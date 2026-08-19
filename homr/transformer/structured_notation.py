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

from dataclasses import dataclass
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


class TieState(StrEnum):
    """Whether this note is joined to its neighbour by a tie.

    Separate from the slur slots on purpose. A tie and a slur are drawn almost alike and
    homr's token vocabulary collapses them - `<tied>` and `<slur>` both emit
    `slurStart`/`slurStop`, so a tie is indistinguishable from a slur in a label file.
    They are different objects: a tie joins two notatons of *one* pitch into a single
    sounding note, while a slur groups distinct pitches under one phrase. Ties are 23% of
    all slur-like markings in this corpus, and 741 notes in an 800-segment sample carry
    both at once, so the conflation is not a rare edge.

    Unlike slurs this needs no slot: a tie joins one pitch to the same pitch, so two ties
    cannot be open on one note of a single voice without being the same tie.
    """

    NONE = "none"
    START = "start"
    STOP = "stop"
    #: A note in the middle of a chain of tied notes both ends one and begins the next.
    START_AND_STOP = "start_and_stop"


class DynamicMark(StrEnum):
    """A dynamics marking attached to a note.

    Values are the MusicXML <dynamics> child element tag names (`p`, `sfz`,
    `other-dynamics`, ...), which is also MuseScore's own DynamicType vocabulary
    (engraving/types/types.h) - the two agree because MuseScore reads/writes MusicXML the
    same way. Using the full ~33-mark set rather than only the handful the OSSQ corpus
    happens to contain follows STEM_CLASSES' DOUBLE precedent: an unused logit costs
    almost nothing, where a class missing from the head's vocabulary cannot be added later
    without invalidating every checkpoint trained against it - and the Lieder corpus this
    design targets next is expected to use marks OSSQ's string quartets do not.
    """

    #: No dynamic attaches to this note - the common case, and a real prediction like
    #: TieState.NONE, not an absence of one.
    NONE = "none"
    #: MusicXML's own catch-all, and also where an unrecognised or hybrid label (two
    #: dynamics children concatenated into one tag, e.g. `pother-dynamics`) collapses to,
    #: per 27.97's finding that these are a data artefact rather than a distinct mark.
    OTHER = "other-dynamics"
    PPPPPP = "pppppp"
    PPPPP = "ppppp"
    PPPP = "pppp"
    PPP = "ppp"
    PP = "pp"
    P = "p"
    MP = "mp"
    MF = "mf"
    F = "f"
    FF = "ff"
    FFF = "fff"
    FFFF = "ffff"
    FFFFF = "fffff"
    FFFFFF = "ffffff"
    FP = "fp"
    PF = "pf"
    SF = "sf"
    SFZ = "sfz"
    SFF = "sff"
    SFFZ = "sffz"
    SFFF = "sfff"
    SFFFZ = "sfffz"
    SFP = "sfp"
    SFPP = "sfpp"
    RFZ = "rfz"
    RF = "rf"
    FZ = "fz"
    M = "m"
    R = "r"
    S = "s"
    Z = "z"
    N = "n"


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


#: Classes the beam-level heads predict. Every state is reachable: FLAG and
#: NOT_APPLICABLE are ordinary predictions, not sentinels.
BEAM_LEVEL_CLASSES: tuple[BeamLevelState, ...] = tuple(BeamLevelState)

#: Classes the stem head predicts. UNKNOWN is excluded on purpose - it is a dataset-side
#: marker for a source that does not say, masked out of the loss, and predicting it would
#: let a silent source be learned as a real answer. DOUBLE is kept despite having no
#: support in OSSQ: an unused logit costs almost nothing, where adding a class later
#: changes the head's vocabulary and invalidates checkpoints trained against it.
STEM_CLASSES: tuple[StemDirection, ...] = tuple(
    state for state in StemDirection if state != StemDirection.UNKNOWN
)

#: Classes a tie head would predict, if one is ever trained. Declared here so the label
#: schema and any future head cannot disagree about the vocabulary.
TIE_CLASSES: tuple[TieState, ...] = tuple(TieState)

#: Classes the dynamics head predicts. Every state is reachable, including NONE - see
#: DynamicMark's docstring for why the vocabulary is the full MusicXML/MuseScore set
#: rather than only OSSQ's observed labels.
DYNAMIC_CLASSES: tuple[DynamicMark, ...] = tuple(DynamicMark)

#: MusicXML <dynamics> child tag -> DynamicMark, for every tag the enum recognises.
_DYNAMIC_TAG_VALUES = {mark.value for mark in DynamicMark if mark != DynamicMark.NONE}


def dynamic_mark_from_tag(tag: str | None) -> DynamicMark:
    """The mark a raw MusicXML <dynamics> tag name maps to.

    None (no direction seen) maps to NONE, the "nothing attached" state. Any tag not in
    the recognised set - an unknown mark, or a hybrid concatenation of two dynamics
    children in one element (27.97's `pother-dynamics`) - maps to OTHER rather than
    raising, since a label pipeline that refuses on an unfamiliar mark would lose real,
    attachable data to a corpus quirk in the source's own text concatenation.
    """
    if tag is None:
        return DynamicMark.NONE
    return DynamicMark(tag) if tag in _DYNAMIC_TAG_VALUES else DynamicMark.OTHER

SLUR_EVENT_CLASSES: tuple[SlurEvent, ...] = tuple(SlurEvent)
SLUR_SIDE_CLASSES: tuple[SlurSide, ...] = tuple(SlurSide)

#: Beam levels given a trained head. The corpus audit found levels 5 and 6 occur 14 times
#: each, all of them in the test split, so a head for either would train on nothing and be
#: evaluated on 14 samples. They stay representable in labels and unsupported by the model.
TRAINED_BEAM_LEVELS = 4

#: Slur slots given trained heads. Slots 3-6 hold 116 training occurrences between them
#: and none in validation above slot 3; slots 1 and 2 hold 291,683 and 2,379. Spans beyond
#: the trained slots are reported as overflow rather than silently unlabelled - and the
#: extractor found zero overflow across the corpus at a cap of six.
TRAINED_SLUR_SLOTS = 2


@dataclass(frozen=True)
class NoteNotation:
    """Structured notation for one note.

    Lives here rather than with the MusicXML extraction because it travels with the
    symbol through the token pipeline, and homr's inference side must be able to name it
    without depending on the training package.
    """

    beam_levels: tuple[BeamLevelState, ...]
    stem: StemDirection
    slurs: tuple[tuple[SlurEvent, SlurSide], ...]
    #: Defaulted so every existing construction stays valid, and so a sidecar written
    #: before ties were extracted decodes as "no tie recorded" rather than failing.
    tie: TieState = TieState.NONE
    #: Defaulted for the same reason as `tie` - a sidecar written before dynamics were
    #: extracted decodes as "no dynamic recorded" rather than failing.
    dynamic: DynamicMark = DynamicMark.NONE

    def active_beam_levels(self) -> int:
        return sum(1 for state in self.beam_levels if state != BeamLevelState.NOT_APPLICABLE)
