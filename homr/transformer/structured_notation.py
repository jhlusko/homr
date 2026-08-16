"""
Schema for the notation the structured heads predict: beams, stem direction, slurs.

Kept separate from the existing token vocabulary on purpose. These are new output-only
dimensions, and folding them into the rhythm vocabulary would rewrite the most important
softmax and embedding matrices in the pretrained model and entangle notation fidelity
with note/rest sequence accuracy.

The representation is per rhythmic beam level rather than a chord-local mode. MuseScore's
BeamMode (AUTO/NONE/BEGIN/BEGIN16/.../END) is a derived editor vocabulary: its MusicXML
importer folds forward and backward hooks and unfamiliar combinations into AUTO, so
training on it would discard exactly the visible information these heads exist to
recover. One state per level maps directly onto <beam number="N"> instead.

Sizing note. The corpus audit (training/omr_datasets/ossq_label_audit.py) found beam
levels 5 and 6 occur 14 times each, all of them in the test split, and slur slots 3-6
occur 116 times in training between them. The caps below are what the *representation*
supports, not a claim about what can be learned; which levels and slots get a trained
head is a separate decision that the support tables drive.
"""

from enum import StrEnum

#: Levels the representation carries: eighth, 16th, 32nd, 64th, 128th, 256th.
MAX_BEAM_LEVELS = 6

#: Concurrent slur slots the representation carries. OSSQ uses slur numbers up to 6.
MAX_SLUR_SLOTS = 6


class BeamLevelState(StrEnum):
    """One rhythmic beam level's state on one note."""

    #: The duration has fewer flags than this level, so the level does not apply.
    NOT_APPLICABLE = "not_applicable"
    #: The level applies visually but is not joined to a neighbour - a flag, not a beam.
    FLAG = "flag"
    BEGIN = "begin"
    CONTINUE = "continue"
    END = "end"
    FORWARD_HOOK = "forward_hook"
    BACKWARD_HOOK = "backward_hook"


class StemDirection(StrEnum):
    """Actual stem direction, independent of beam connectivity."""

    NOT_APPLICABLE = "not_applicable"
    UP = "up"
    DOWN = "down"
    NONE = "none"
    DOUBLE = "double"
    #: Dataset-side sentinel for a source that does not say. Masked out of the loss and
    #: never an inference class, so a silent source cannot be learned as a real answer.
    UNKNOWN = "unknown"


class SlurEvent(StrEnum):
    NONE = "none"
    START = "start"
    STOP = "stop"
    #: A note can close one span and open another in the same canonical slot.
    START_AND_STOP = "start_and_stop"
    CONTINUE = "continue"


class SlurSide(StrEnum):
    #: The majority of slurs in the corpus carry no placement at all, so this is the
    #: common case rather than an edge case, and direction loss only applies where the
    #: source is explicit.
    UNSPECIFIED = "unspecified"
    ABOVE = "above"
    BELOW = "below"


#: MusicXML <type> -> number of flags, i.e. the highest beam level that applies.
_FLAGS_BY_NOTE_TYPE = {
    "maxima": 0,
    "long": 0,
    "breve": 0,
    "whole": 0,
    "half": 0,
    "quarter": 0,
    "eighth": 1,
    "16th": 2,
    "32nd": 3,
    "64th": 4,
    "128th": 5,
    "256th": 6,
    "512th": 7,
    "1024th": 8,
}


def applicable_beam_levels(note_type: str | None) -> int:
    """How many beam levels apply to this MusicXML note type.

    Returns 0 for durations that carry no flags and for an unrecognised or missing type,
    which makes every level NOT_APPLICABLE rather than inventing supervision for a note
    whose duration we could not read.
    """
    if note_type is None:
        return 0
    return min(_FLAGS_BY_NOTE_TYPE.get(note_type.strip(), 0), MAX_BEAM_LEVELS)


def empty_beam_levels() -> tuple[BeamLevelState, ...]:
    return (BeamLevelState.NOT_APPLICABLE,) * MAX_BEAM_LEVELS


def empty_slur_slots() -> tuple[tuple[SlurEvent, SlurSide], ...]:
    return ((SlurEvent.NONE, SlurSide.UNSPECIFIED),) * MAX_SLUR_SLOTS
