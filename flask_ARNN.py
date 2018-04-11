import tempfile

from ARNN.anticipationRNN import AnticipationRNN
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import TickMetadata
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from music21 import musicxml, metadata, converter
import click

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'midi'}

# INITIALIZATION
xml_response_headers = {"Content-Type": "text/xml",
                        "charset":      "utf-8"
                        }

_current_tensor_chorale = None
_current_tensor_metadata = None
_current_chorale = None
model = None

# generation parameters
batch_size_per_voice = 8


@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.2,
              help='amount of dropout between LSTM layers')
@click.option('--input_dropout', default=0.2,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
def init_app(note_embedding_dim,
             meta_embedding_dim,
             num_layers,
             lstm_hidden_size,
             dropout_lstm,
             input_dropout,
             linear_hidden_size,
             ):
    metadatas = [
        TickMetadata(subdivision=4),
    ]

    dataset_manager = DatasetManager()
    chorale_dataset_kwargs = {
        'voice_ids':      [0],
        'metadatas':      metadatas,
        'sequences_size': 20,
        'subdivision':    4
    }

    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
    )

    global model
    model = AnticipationRNN(chorale_dataset=bach_chorales_dataset,
                            note_embedding_dim=note_embedding_dim,
                            metadata_embedding_dim=meta_embedding_dim,
                            num_layers=num_layers,
                            num_lstm_constraints_units=lstm_hidden_size,
                            num_lstm_generation_units=lstm_hidden_size,
                            linear_hidden_size=linear_hidden_size,
                            dropout_prob=dropout_lstm,
                            dropout_input_prob=input_dropout,
                            unary_constraint=True,
                            )
    model.load()
    model.cuda()

    # launch the script
    # accessible only locally:
    app.run()


@app.route('/test', methods=['POST', 'GET'])
def test_generation():
    response = make_response(('TEST', xml_response_headers))

    if request.method == 'POST':
        print(request)

    return response


@app.route('/models', methods=['GET'])
def models():
    return jsonify(['ARNN'])


@app.route('/compose', methods=['POST'])
def compose():
    global _current_tensor_chorale
    global _current_tensor_metadata
    global _current_chorale
    global model

    # global models
    # --- Parse request---
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml') as file:
        print(file.name)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        xml_string = request.form['xml_string']
        file.write(xml_string)

        # load chorale with music21
        _current_chorale = converter.parse(file.name)

        NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE = 120
        start_tick_selection = int(float(
            request.form['start_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE)
        end_tick_selection = int(
            float(request.form['end_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE)

        # use length of the score shown in MuseScore
        chorale_length = int(_current_chorale.duration.quarterLength
                             * model.chorale_dataset.subdivision)
        # if no selection REGENERATE and set chorale length
        if start_tick_selection == 0 and end_tick_selection == 0:
            # randomize
            _current_tensor_chorale = model.chorale_dataset.random_chorale(
                chorale_length=chorale_length)
            _current_tensor_metadata = None

            end_tick_selection = chorale_length
        else:
            _current_tensor_chorale = model.chorale_dataset.chorale_to_tensor(
                chorale=_current_chorale,
                offsetStart=0,
                offsetEnd=chorale_length // model.chorale_dataset.subdivision
            )

        time_index_range = [start_tick_selection, end_tick_selection]


        # --- Generate---
        (_current_chorale,
         _current_tensor_chorale,
         _current_tensor_metadata) = model.generation(
            tensor_chorale=_current_tensor_chorale,
            tensor_metadata=_current_tensor_metadata,
            temperature=1.,
            time_index_range=time_index_range
        )

        # format metadatadata
        insert_metadata(_current_chorale)

        # convert chorale to xml
        response = chorale_to_xml_response(_current_chorale)
    return response


def insert_metadata(output_chorale):
    for part, name in zip(output_chorale.parts, ['soprano']):
        part.id = name
        part.partName = name
    md = metadata.Metadata()
    output_chorale.insert(0, md)
    output_chorale.metadata.title = 'Anticipation-RNN'
    output_chorale.metadata.composer = 'GH'


def parse_request(req):
    """
    must cast
    :param req:
    :return:
    """
    measure_index = req.args.get('measureIndex')
    if measure_index is not None:
        measure_index = int(req.args.get('measureIndex'))

    return {'measure_index': measure_index
            }


def chorale_to_xml_response(chorale):
    goe = musicxml.m21ToXml.GeneralObjectExporter(chorale)
    xml_chorale_string = goe.parse()

    response = make_response((xml_chorale_string, xml_response_headers))
    return response


if __name__ == '__main__':
    init_app()
