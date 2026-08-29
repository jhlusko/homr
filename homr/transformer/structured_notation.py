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
    #: Sentinel for "this decoder position was never supervised" (padding, BOS/EOS, a
    #: non-note token) - never a real label, never an inference class (excluded from
    #: TIE_CLASSES below, mirroring StemDirection.UNKNOWN). Distinct from NONE, which IS
    #: a real, scoreable prediction ("this note is plainly not tied"). Before this
    #: existed, `structured_decoding.py::_lookup` decoded a masked position to NONE - the
    #: same class as a real "not tied" answer - so `tie_report`/`dynamic_report` scored
    #: every padding/BOS/EOS/non-note position as a free correct prediction. Measured
    #: impact: tie's `none` support was inflated 9.2x (607 x n_sequences instead of the
    #: true target count); because tie's real none-accuracy is already ~0.998 the
    #: distortion was small there, but the same mechanism creates a ~0.10 macro-F1 floor
    #: for the dynamics head, whose none-accuracy is nowhere near that high - see
    #: docs/private/DYNAMICS_HEAD_FINDINGS.md.
    UNKNOWN = "unknown"


class AdvanceClass(StrEnum):
    """How much time passes before the NEXT simultaneity in this staff/voice.

    Exists because the renderer's fallback rule - a simultaneity's duration is the
    MINIMUM among its members (`SymbolChord.get_duration`) - is exact only when the two
    hands of a grand staff share every onset. Measured on the rebuilt Lieder corpus:
    25.5% of grand-staff simultaneities hold notes of different lengths (46.4% where both
    hands sound at once), and on 10.3% of grand-staff bars the min-rule's own total
    disagrees with the bar's own modal length. That is not recoverable from the label
    after the fact - see `training.omr_datasets.staff_merging` for where it is still
    available and computed. See docs/private/ONSET_REPRESENTATION_RESEARCH.md.

    Values name a duration exactly like a rhythm token's own value string (`4`, `8.`,
    ...), deliberately reusing that vocabulary rather than inventing a second one -
    `homr.tuplet_repair.duration` parses both alike.
    """

    #: No question was asked here: not the canonical (last) symbol of a simultaneity, not
    #: a note-bearing symbol at all, or the last simultaneity of its measure with no
    #: following one available to compute a delta against (see the module docstring in
    #: `staff_merging.py` for why a delta is not carried across a measure boundary).
    NOT_APPLICABLE = "not_applicable"
    #: The next simultaneity starts at the SAME true onset as this one - typically a
    #: grace note or an inserted attribute change sharing a moment with a real note. A
    #: real, informative answer ("nothing to wait for"), not an absence of one.
    ZERO = "zero"
    #: A real, nonzero gap that does not quantize exactly to a notated duration - most
    #: often a tuplet-derived ratio the fixed class list below does not carry a slot for.
    OTHER = "other"
    WHOLE = "1"
    DOTTED_HALF = "2."
    HALF = "2"
    DOTTED_QUARTER = "4."
    QUARTER = "4"
    DOTTED_EIGHTH = "8."
    EIGHTH = "8"
    DOTTED_16TH = "16."
    SIXTEENTH = "16"
    DOTTED_32ND = "32."
    THIRTY_SECOND = "32"
    DOTTED_64TH = "64."
    SIXTY_FOURTH = "64"


#: Classes the advance head predicts. Declared here, not derived from a "trained subset"
#: the way DYNAMIC_CLASSES is, because the vocabulary is already small (15 classes) and
#: every value is meaningful on its own - there is no long tail of rare tags to fold away.
ADVANCE_CLASSES: tuple[AdvanceClass, ...] = tuple(AdvanceClass)


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

    This is the representation's vocabulary, not necessarily a trained head's: see
    `TRAINED_DYNAMIC_MARKS` for the smaller set phase16 (28.1) found any one training run
    can actually make use of. `NoteNotation.dynamic` and the sidecar always carry the full
    mark; only `structured_targets.py`'s target-building collapses it.
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
    #: Sentinel for "this decoder position was never supervised" - see
    #: `TieState.UNKNOWN`'s docstring for the full rationale (same bug, same fix, applied
    #: here too). Never in `TRAINED_DYNAMIC_MARKS`, so it is automatically excluded from
    #: `DYNAMIC_CLASSES` - never a real label, never an inference class.
    UNKNOWN = "unknown"


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

#: Classes the tie head predicts. UNKNOWN excluded for the same reason STEM_CLASSES
#: excludes it - a masked-position sentinel, never a real label. Filtering (not
#: reordering) the first four `TieState` members leaves their indices unchanged, so this
#: is safe against every checkpoint already trained on the 4-class TIE_CLASSES.
TIE_CLASSES: tuple[TieState, ...] = tuple(state for state in TieState if state != TieState.UNKNOWN)

#: MusicXML <dynamics> child tag -> DynamicMark, for every tag the enum recognises.
_DYNAMIC_TAG_VALUES = {mark.value for mark in DynamicMark if mark != DynamicMark.NONE}


def dynamic_mark_from_tag(tag: str | None) -> DynamicMark:
    """The mark a raw MusicXML <dynamics> tag name maps to.

    None (no direction seen) maps to NONE, the "nothing attached" state. Any tag not in
    the recognised set - an unknown mark, or a hybrid concatenation of two dynamics
    children in one element (27.97's `pother-dynamics`) - maps to OTHER rather than
    raising, since a label pipeline that refuses on an unfamiliar mark would lose real,
    attachable data to a corpus quirk in the source's own text concatenation.

    This is the full-fidelity mapping `NoteNotation.dynamic` and the sidecar carry - see
    `TRAINED_DYNAMIC_MARKS` for the separate, smaller vocabulary a trained head predicts.
    """
    if tag is None:
        return DynamicMark.NONE
    return DynamicMark(tag) if tag in _DYNAMIC_TAG_VALUES else DynamicMark.OTHER


#: Marks given a trained head, mirroring TRAINED_BEAM_LEVELS/TRAINED_SLUR_SLOTS: the
#: representation (DynamicMark, and what NoteNotation.dynamic/the sidecar carry) stays the
#: full ~33-tag set, independent of what any one training run asks the head to
#: discriminate. 27.96/27.97's corpus audit found p/f/mf/pp/sf/ff/mp/ppp cover ~97% of
#: examples; phase16 (28.1) measured that the rest have too few examples for loss
#: reweighting to help regardless - nine of the seventeen marks observed in one eval had
#: single- or low-double-digit support, and scoped focal loss (which fixed the tie head)
#: left dynamics' macro F1 flat. Folding everything else to OTHER concentrates what little
#: signal exists onto marks the head has already shown it can partly learn, without
#: shrinking what a sidecar can represent - Lieder may yet make a folded mark common
#: enough to add back, which is why this is a target-time collapse, not a schema change.
TRAINED_DYNAMIC_MARKS: frozenset[DynamicMark] = frozenset(
    {
        DynamicMark.NONE,
        DynamicMark.OTHER,
        DynamicMark.P,
        DynamicMark.F,
        DynamicMark.MF,
        DynamicMark.PP,
        DynamicMark.SF,
        DynamicMark.FF,
        DynamicMark.MP,
        DynamicMark.PPP,
    }
)


def trained_dynamic_mark(mark: DynamicMark) -> DynamicMark:
    """Collapse a mark to the head's trained vocabulary, folding everything else to OTHER."""
    return mark if mark in TRAINED_DYNAMIC_MARKS else DynamicMark.OTHER


#: Classes the dynamics head predicts - the trained subset, not the full representation.
#: Order follows DynamicMark's own declaration order so index assignment is stable.
DYNAMIC_CLASSES: tuple[DynamicMark, ...] = tuple(
    mark for mark in DynamicMark if mark in TRAINED_DYNAMIC_MARKS
)

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
    #: Defaulted for the same reason as `tie` and `dynamic` - a sidecar written before
    #: advances were extracted decodes as NOT_APPLICABLE, which is also the correct
    #: value on every symbol that is not the last of its simultaneity (see
    #: `staff_merging.create_chord_over_two_staffs`), so an old sidecar and a new one
    #: agree on every position they don't both speak to.
    advance: AdvanceClass = AdvanceClass.NOT_APPLICABLE

    def active_beam_levels(self) -> int:
        return sum(1 for state in self.beam_levels if state != BeamLevelState.NOT_APPLICABLE)
