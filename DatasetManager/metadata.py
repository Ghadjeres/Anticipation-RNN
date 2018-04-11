"""
Metadata classes
"""
import numpy as np
from music21 import analysis, stream


class Metadata:
    def __init__(self):
        self.num_values = None
        self.is_global = None
        self.name = None

    def get_index(self, value):
        # trick with the 0 value
        raise NotImplementedError

    def get_value(self, index):
        raise NotImplementedError

    def evaluate(self, chorale, subdivision):
        """
        takes a music21 chorale as input and the number of subdivisions per beat
        """
        raise NotImplementedError

    def generate(self, length):
        raise NotImplementedError


class TickMetadata(Metadata):
    """
    Metadata class that tracks on which subdivision of the beat we are on
    """

    def __init__(self, subdivision):
        super(TickMetadata, self).__init__()
        self.is_global = False
        self.num_values = subdivision
        self.name = 'tick'

    def get_index(self, value):
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale, subdivision):
        assert subdivision == self.num_values
        # suppose all pieces start on a beat
        length = int(chorale.duration.quarterLength * subdivision)
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))

    def generate(self, length):
        return np.array(list(map(
            lambda x: x % self.num_values,
            range(length)
        )))


class KeyMetadata(Metadata):
    """
    Metadata class that indicates in which key we are
    Only returns the number of sharps or flats
    Does not distinguish a key from its relative key
    """

    def __init__(self, window_size=4):
        super(KeyMetadata, self).__init__()
        self.window_size = window_size
        self.is_global = False
        self.num_max_sharps = 7
        self.num_values = 16
        self.name = 'key'

    def get_index(self, value):
        """

        :param value: number of sharps (between -7 and +7)
        :return: index in the representation
        """
        return value + self.num_max_sharps + 1

    def get_value(self, index):
        """

        :param index:  index (between 0 and self.num_values); 0 is unused (no constraint)
        :return: true number of sharps (between -7 and 7)
        """
        return index - 1 - self.num_max_sharps

    def evaluate(self, chorale, subdivision):
        # init key analyzer
        # we must add measures by hand for the case when we are parsing midi files
        chorale_with_measures = stream.Score()
        for part in chorale.parts:
            chorale_with_measures.append(part.makeMeasures())

        ka = analysis.floatingKey.KeyAnalyzer(chorale_with_measures)
        ka.windowSize = self.window_size
        res = ka.run()

        measure_offset_map = chorale_with_measures.parts.measureOffsetMap()
        length = int(chorale.duration.quarterLength * subdivision)  # in 16th notes

        key_signatures = np.zeros((length,))

        measure_index = -1
        for time_index in range(length):
            beat_index = time_index / subdivision
            if beat_index in measure_offset_map:
                measure_index += 1
                if measure_index == len(res):
                    measure_index -= 1

            key_signatures[time_index] = self.get_index(res[measure_index].sharps)
        return np.array(key_signatures, dtype=np.int32)

    def generate(self, length):
        return np.full((length,), self.get_index(0))


class FermataMetadata(Metadata):
    """
    Metadata class which indicates if a fermata is on the current note
    """

    def __init__(self):
        super(FermataMetadata, self).__init__()
        self.is_global = False
        self.num_values = 2
        self.name = 'fermata'

    def get_index(self, value):
        # values are 1 and 0
        return value

    def get_value(self, index):
        return index

    def evaluate(self, chorale, subdivision):
        part = chorale.parts[0]
        length = int(part.duration.quarterLength * subdivision)  # in 16th notes
        list_notes = part.flat.notes
        num_notes = len(list_notes)
        j = 0
        i = 0
        fermatas = np.zeros((length,))
        while i < length:
            if j < num_notes - 1:
                if list_notes[j + 1].offset > i / subdivision:

                    if len(list_notes[j].expressions) == 1:
                        fermata = True
                    else:
                        fermata = False
                    fermatas[i] = fermata
                    i += 1
                else:
                    j += 1
            else:
                if len(list_notes[j].expressions) == 1:
                    fermata = True
                else:
                    fermata = False

                fermatas[i] = fermata
                i += 1
        return np.array(fermatas, dtype=np.int32)

    def generate(self, length):
        # fermata every 2 bars
        return np.array([1 if i % 32 >= 28 else 0
                         for i in range(length)])
