import music21
from itertools import islice

from music21 import note, harmony, expressions

# constants
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'


def standard_name(note_or_rest):
    """
    Convert music21 objects to str
    :param note_or_rest:
    :return:
    """
    if isinstance(note_or_rest, note.Note):
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, note.Rest):
        return note_or_rest.name
    if isinstance(note_or_rest, str):
        return note_or_rest
    if isinstance(note_or_rest, harmony.ChordSymbol):
        return note_or_rest.figure
    if isinstance(note_or_rest, expressions.TextExpression):
        return note_or_rest.content


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    if note_or_rest_string == START_SYMBOL or note_or_rest_string == END_SYMBOL:
        return note.Rest()
    if note_or_rest_string == SLUR_SYMBOL:
        print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


class ShortChoraleIteratorGen:
    """
    Class used for debugging
    when called, it returns an iterator over 3 Bach chorales,
    similar to music21.corpus.chorales.Iterator()
    """

    def __init__(self):
        pass

    def __call__(self):
        it = (
            chorale
            for chorale in
            islice(music21.corpus.chorales.Iterator(), 3)
        )
        return it.__iter__()
