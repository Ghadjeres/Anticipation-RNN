import click

from ARNN.constraint_sets import C3
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import TickMetadata
from ARNN.anticipationRNN import AnticipationRNN


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
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=5,
              help='number of training epochs')
@click.option('--train', is_flag=True,
              help='train the specified model')
@click.option('--no_metadata', is_flag=True,
              help='do not use metadata')
def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         input_dropout,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         no_metadata,
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
                            no_metadata=no_metadata,
                            )

    if train:
        model.cuda()
        model.train_model(batch_size=batch_size,
                          num_epochs=num_epochs
                          )
    else:
        model.load()
        model.cuda()

    print('Fill')
    score, _, _ = model.fill(C3)
    score.show()


if __name__ == '__main__':
    main()
