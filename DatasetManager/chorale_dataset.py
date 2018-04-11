import music21
import torch
import numpy as np

from music21 import interval, stream
from torch.utils.data import TensorDataset
from tqdm import tqdm

from DatasetManager.helpers import standard_name, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, \
    standard_note
from DatasetManager.metadata import FermataMetadata
from DatasetManager.music_dataset import MusicDataset


class ChoraleDataset(MusicDataset):
    """
    Class for all chorale-like datasets
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 voice_ids,
                 metadatas=None,
                 sequences_size=8,
                 subdivision=4,
                 cache_dir=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param voice_ids: list of voice_indexes to be used
        :param metadatas: list[Metadata]
        :param sequences_size: in beats
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super(ChoraleDataset, self).__init__(cache_dir=cache_dir)
        self.voice_ids = voice_ids
        self.num_voices = len(voice_ids)
        self.name = name
        self.sequences_size = sequences_size
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.chorale_corpus_it_gen = corpus_it_gen
        self.voice_ranges = None  # in midi pitch
        self.metadatas = metadatas
        self.subdivision = subdivision

    def __repr__(self):
        return f'ChoraleDataset(' \
               f'{self.voice_ids},' \
               f'{self.name},' \
               f'{[metadata.name for metadata in self.metadatas]},' \
               f'{self.sequences_size},' \
               f'{self.subdivision})'

    def chorale_iterator_gen(self):
        return (chorale
                for chorale in self.chorale_corpus_it_gen()
                if self.is_valid(chorale)
                )

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')
        self.compute_index_dicts()
        self.compute_voice_ranges()
        one_tick = 1 / self.subdivision
        chorale_tensor_dataset = []
        metadata_tensor_dataset = []
        for chorale_id, chorale in tqdm(enumerate(self.chorale_iterator_gen())):

            # precompute all possible transpositions and corresponding metadatas
            chorale_transpositions = {}
            metadatas_transpositions = {}

            # main loop
            for offsetStart in np.arange(
                    chorale.flat.lowestOffset -
                    (self.sequences_size - one_tick),
                    chorale.flat.highestOffset,
                    one_tick):
                offsetEnd = offsetStart + self.sequences_size
                current_subseq_ranges = self.voice_range_in_subsequence(
                    chorale,
                    offsetStart=offsetStart,
                    offsetEnd=offsetEnd)

                transposition = self.min_max_transposition(current_subseq_ranges)
                min_transposition_subsequence, max_transposition_subsequence = transposition

                for semi_tone in range(min_transposition_subsequence,
                                       max_transposition_subsequence + 1):
                    start_tick = int(offsetStart * self.subdivision)
                    end_tick = int(offsetEnd * self.subdivision)

                    try:
                        # compute transpositions lazily
                        if semi_tone not in chorale_transpositions:
                            (chorale_tensor,
                             metadata_tensor) = self.transposed_chorale_and_metadata_tensors(
                                chorale,
                                semi_tone=semi_tone)
                            chorale_transpositions.update(
                                {semi_tone:
                                     chorale_tensor})
                            metadatas_transpositions.update(
                                {semi_tone:
                                     metadata_tensor})
                        else:
                            chorale_tensor = chorale_transpositions[semi_tone]
                            metadata_tensor = metadatas_transpositions[semi_tone]

                        local_chorale_tensor = self.extract_chorale_with_padding(
                            chorale_tensor,
                            start_tick, end_tick)
                        local_metadata_tensor = self.extract_metadata_with_padding(
                            metadata_tensor,
                            start_tick, end_tick)

                        # append and add batch dimension
                        # cast to int
                        chorale_tensor_dataset.append(
                            local_chorale_tensor[None, :, :].int())
                        metadata_tensor_dataset.append(
                            local_metadata_tensor[None, :, :, :].int())
                    except KeyError:
                        # some problems may occur with the key analyzer
                        print(f'KeyError with chorale {chorale_id}')

        chorale_tensor_dataset = torch.cat(chorale_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)

        dataset = TensorDataset(chorale_tensor_dataset,
                                metadata_tensor_dataset)

        print(f'Sizes: {chorale_tensor_dataset.size()}, {metadata_tensor_dataset.size()}')
        return dataset

    def transposed_chorale_and_metadata_tensors(self, chorale, semi_tone):
        """
        Convert chorale to a couple (chorale_tensor, metadata_tensor),
        the original chorale is transposed semi_tone number of semi-tones

        :param chorale: music21 object
        :param semi_tone:
        :return: couple of tensors
        """
        # transpose
        # compute the most "natural" interval given a number of semi-tones
        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(
            semi_tone)
        transposition_interval = interval.Interval(
            str(interval_nature) + interval_type)

        chorale_tranposed = chorale.transpose(transposition_interval)
        chorale_tensor = self.chorale_to_tensor(chorale_tranposed,
                                                offsetStart=0.,
                                                offsetEnd=chorale_tranposed.flat.highestTime)
        metadatas_transposed = self.compute_metadata(chorale_tranposed)
        return chorale_tensor, metadatas_transposed

    def compute_metadata(self, chorale):
        """
        Adds also the index of the voices

        :param chorale: music21 stream
        :return:tensor (num_voices, chorale_length, len(self.metadatas) + 1)
        """
        md = []
        if self.metadatas:
            for metadata in self.metadatas:
                sequence_metadata = torch.from_numpy(
                    metadata.evaluate(chorale, self.subdivision)).long().clone()
                square_metadata = sequence_metadata.repeat(self.num_voices, 1)
                md.append(
                    square_metadata[:, :, None]
                )
        chorale_length = int(chorale.duration.quarterLength * self.subdivision)

        # add voice indexes
        voice_id_metada = torch.from_numpy(np.arange(self.num_voices)).long().clone()
        square_metadata = torch.transpose(voice_id_metada.repeat(chorale_length, 1),
                                          0, 1)
        md.append(square_metadata[:, :, None])

        all_metadata = torch.cat(md, 2)
        return all_metadata

    def add_fermata(self, metadata_tensor, time_index_start,
                    time_index_stop):
        if self.metadatas:
            for metadata_index, metadata in enumerate(self.metadatas):
                if isinstance(metadata, FermataMetadata):
                    metadata_tensor[:,
                    time_index_start: time_index_stop,
                    metadata_index] = 1
            return metadata_tensor
        else:
            return metadata_tensor

    def min_max_transposition(self, current_subseq_ranges):
        if current_subseq_ranges is None:
            # there is no note in one part
            transposition = (0, 0)  # min and max transpositions
        else:
            transpositions = [
                (min_pitch_corpus - min_pitch_current,
                 max_pitch_corpus - max_pitch_current)
                for ((min_pitch_corpus, max_pitch_corpus),
                     (min_pitch_current, max_pitch_current))
                in zip(self.voice_ranges, current_subseq_ranges)
            ]
            transpositions = [min_or_max_transposition
                              for min_or_max_transposition in zip(*transpositions)]
            transposition = [max(transpositions[0]),
                             min(transpositions[1])]
        return transposition

    def chorale_to_tensor(self, chorale, offsetStart, offsetEnd):
        chorale_tensor = []
        for part_id, part in enumerate(chorale.parts[:self.num_voices]):
            part_tensor = self.part_to_tensor(part, part_id,
                                              offsetStart=offsetStart,
                                              offsetEnd=offsetEnd)
            chorale_tensor.append(part_tensor)
        return torch.cat(chorale_tensor, 0)

    def part_to_tensor(self, part, part_id, offsetStart, offsetEnd):
        """

        :param part:
        :param part_id:
        :param offsetStart:
        :param offsetEnd:
        :return: torch IntTensor (1, length)
        """
        list_notes_and_rests = list(part.flat.getElementsByOffset(
            offsetStart=offsetStart,
            offsetEnd=offsetEnd,
            classList=[music21.note.Note,
                       music21.note.Rest]))
        list_note_strings = [n.nameWithOctave for n in list_notes_and_rests
                             if n.isNote]
        length = int((offsetEnd - offsetStart) * self.subdivision)  # in ticks

        # add entries to dictionaries if not present
        # should only be called by make_dataset when transposing
        for note_name in list_note_strings:
            note2index = self.note2index_dicts[part_id]
            index2note = self.index2note_dicts[part_id]
            if note_name not in note2index:
                new_index = len(note2index)
                index2note.update({new_index: note_name})
                note2index.update({note_name: new_index})
                print('Warning: Entry ' + str(
                    {new_index: note_name}) + ' added to dictionaries')

        # construct sequence
        j = 0
        i = 0
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(list_notes_and_rests)
        while i < length:
            if j < num_notes - 1:
                if list_notes_and_rests[j + 1].offset > i / self.subdivision + offsetStart:
                    t[i, :] = [note2index[standard_name(list_notes_and_rests[j])],
                               is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(list_notes_and_rests[j])],
                           is_articulated]
                i += 1
                is_articulated = False
        seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        tensor = torch.from_numpy(seq).long()[None, :]
        return tensor

    def voice_range_in_subsequence(self, chorale, offsetStart, offsetEnd):
        """
        returns None if no note present in one of the voices -> no transposition
        :param chorale:
        :param offsetStart:
        :param offsetEnd:
        :return:
        """
        voice_ranges = []
        for part in chorale.parts[:self.num_voices]:
            voice_range_part = self.voice_range_in_part(part,
                                                        offsetStart=offsetStart,
                                                        offsetEnd=offsetEnd)
            if voice_range_part is None:
                return None
            else:
                voice_ranges.append(voice_range_part)
        return voice_ranges

    def voice_range_in_part(self, part, offsetStart, offsetEnd):
        notes_in_subsequence = part.flat.getElementsByOffset(
            offsetStart,
            offsetEnd,
            includeEndBoundary=False,
            mustBeginInSpan=True,
            mustFinishInSpan=False,
            classList=[music21.note.Note,
                       music21.note.Rest])
        midi_pitches_part = [
            n.pitch.midi
            for n in notes_in_subsequence
            if n.isNote
        ]
        if len(midi_pitches_part) > 0:
            return min(midi_pitches_part), max(midi_pitches_part)
        else:
            return None

    def compute_index_dicts(self):
        print('Computing index dicts')
        self.index2note_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.note2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)

        # get all notes
        for chorale in tqdm(self.chorale_iterator_gen()):
            for part_id, part in enumerate(chorale.parts[:self.num_voices]):
                for n in part.flat.notesAndRests:
                    note_sets[part_id].add(standard_name(n))

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2note_dicts,
                                                    self.note2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})

    def is_valid(self, chorale):
        # We only consider 4-part chorales
        if not len(chorale.parts) == 4:
            return False
        return True

    def compute_voice_ranges(self):
        assert self.index2note_dicts is not None
        assert self.note2index_dicts is not None
        self.voice_ranges = []
        print('Computing voice ranges')
        for voice_index, note2index in tqdm(enumerate(self.note2index_dicts)):
            notes = [
                standard_note(note_string)
                for note_string in note2index
            ]
            midi_pitches = [
                n.pitch.midi
                for n in notes
                if n.isNote
            ]
            min_midi, max_midi = min(midi_pitches), max(midi_pitches)
            self.voice_ranges.append((min_midi, max_midi))

    def extract_chorale_with_padding(self, tensor_chorale, start_tick, end_tick):
        """

        :param tensor_chorale: (num_voices, length in ticks)
        :param start_tick:
        :param end_tick:
        :return: tensor_chorale[:, start_tick: end_tick]
        with padding if necessary
        i.e. if start_tick < 0 or end_tick > tensor_chorale length
        """
        assert start_tick < end_tick
        assert end_tick > 0
        length = tensor_chorale.size()[1]

        padded_chorale = []
        if start_tick < 0:
            start_symbols = np.array([note2index[START_SYMBOL]
                                      for note2index in self.note2index_dicts])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(-start_tick, 1).transpose(0, 1)
            padded_chorale.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_chorale.append(tensor_chorale[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([note2index[END_SYMBOL]
                                    for note2index in self.note2index_dicts])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(end_tick - length, 1).transpose(0, 1)
            padded_chorale.append(end_symbols)

        padded_chorale = torch.cat(padded_chorale, 1)
        return padded_chorale

    def extract_metadata_with_padding(self, tensor_metadata,
                                      start_tick, end_tick):
        """

        :param tensor_metadata: (num_voices, length, num_metadatas)
        last metadata is the voice_index
        :param start_tick:
        :param end_tick:
        :return:
        """
        assert start_tick < end_tick
        assert end_tick > 0
        num_voices, length, num_metadatas = tensor_metadata.size()
        padded_tensor_metadata = []

        if start_tick < 0:
            start_symbols = np.zeros((self.num_voices, -start_tick, num_metadatas))
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            padded_tensor_metadata.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length
        padded_tensor_metadata.append(tensor_metadata[:, slice_start: slice_end, :])

        if end_tick > length:
            end_symbols = np.zeros((self.num_voices, end_tick - length, num_metadatas))
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            padded_tensor_metadata.append(end_symbols)

        padded_tensor_metadata = torch.cat(padded_tensor_metadata, 1)
        return padded_tensor_metadata

    def empty_chorale(self, chorale_length):
        start_symbols = np.array([note2index[START_SYMBOL]
                                  for note2index in self.note2index_dicts])
        start_symbols = torch.from_numpy(start_symbols).long().clone()
        start_symbols = start_symbols.repeat(chorale_length, 1).transpose(0, 1)
        return start_symbols

    def random_chorale(self, chorale_length):
        chorale_tensor = np.array(
            [np.random.randint(len(note2index),
                               size=chorale_length)
             for note2index in self.note2index_dicts])
        chorale_tensor = torch.from_numpy(chorale_tensor).long().clone()
        return chorale_tensor

    def tensor_chorale_to_score(self, tensor_chorale):
        """

        :param tensor_chorale: (num_voices, length)
        :return:
        """
        slur_indexes = [note2index[SLUR_SYMBOL]
                        for note2index in self.note2index_dicts]

        score = music21.stream.Score()
        for voice_index, (voice, index2note, slur_index) in enumerate(
                zip(tensor_chorale,
                    self.index2note_dicts,
                    slur_indexes)):
            part = stream.Part(id='part' + str(voice_index))
            dur = 0
            f = music21.note.Rest()
            for note_index in voice:
                # if it is a played note
                if not note_index == slur_indexes[voice_index]:
                    # add previous note
                    if dur > 0:
                        f.duration = music21.duration.Duration(dur / self.subdivision)
                        part.append(f)

                    dur = 1
                    f = standard_note(index2note[note_index])
                else:
                    dur += 1
            # add last note
            f.duration = music21.duration.Duration(dur / self.subdivision)
            part.append(f)
            score.insert(part)
        return score
