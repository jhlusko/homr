"""
What a checkpoint can actually do, recorded so consumers do not have to guess.

A model that has grown heads is not the same as a model whose heads mean anything. The
projections exist from the moment the architecture changes; they produce logits
immediately, and an untrained one produces confident nonsense. Nothing in the weights
distinguishes the two states, so the checkpoint has to say.

That is what this manifest is for, and why declaring a head is a separate act from having
one. `build` takes the heads a run actually trained and refuses to declare any other, so
"supported" means trained rather than present.

The other half is class order. Every head's logits are positional - index 3 of a beam
head means BACKWARD_HOOK because that is where it sits in the enum - so reordering the
classes silently reinterprets every prediction a checkpoint ever made, with no error
anywhere. Hashing each head's ordered class list turns that into a mismatch a consumer
can detect instead of a wrong answer it cannot.

Consumers must treat a head missing from the manifest as unsupported, never as a
confident prediction of NONE.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    DYNAMIC_CLASSES,
    MAX_BEAM_LEVELS,
    MAX_SLUR_SLOTS,
    SLUR_EVENT_CLASSES,
    SLUR_SIDE_CLASSES,
    STEM_CLASSES,
    TIE_CLASSES,
)

SCHEMA_VERSION = "homr.capability-manifest.v1"

#: Structured heads read the hidden state and feed nothing back into the next
#: autoregressive step. Recorded because a later experiment may change it, and a consumer
#: cannot tell from the weights.
STRUCTURED_HEADS_ARE_AUTOREGRESSIVE = False


def _classes_for(head: str) -> tuple[str, ...]:
    if head.startswith("beam.level."):
        return tuple(str(state) for state in BEAM_LEVEL_CLASSES)
    if head == "stem.direction":
        return tuple(str(state) for state in STEM_CLASSES)
    if head == "tie.state":
        return tuple(str(state) for state in TIE_CLASSES)
    if head == "dynamic.mark":
        return tuple(str(state) for state in DYNAMIC_CLASSES)
    if head.endswith(".event"):
        return tuple(str(state) for state in SLUR_EVENT_CLASSES)
    if head.endswith(".side"):
        return tuple(str(state) for state in SLUR_SIDE_CLASSES)
    raise KeyError(f"no class list known for head {head!r}")


def vocabulary_hash(head: str) -> str:
    """Hash of a head's ordered class list; changes if the classes or their order change."""
    payload = json.dumps(_classes_for(head), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class CapabilityManifest:
    model_revision: str
    training_revision: str
    #: Heads this checkpoint was trained for, and may be asked for.
    supported_heads: tuple[str, ...]
    head_vocabulary_hashes: dict[str, str]
    max_beam_levels: int
    max_slur_slots: int
    structured_heads_autoregressive: bool
    max_image_height: int
    max_image_width: int
    max_sequence_length: int
    label_schema_version: str
    dataset_revision: str = ""
    run_id: str = ""
    schema_version: str = SCHEMA_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    def supports(self, head: str) -> bool:
        return head in self.supported_heads

    def check_compatible(self, head: str, classes: tuple[str, ...]) -> None:
        """Raise unless this checkpoint supports `head` with exactly these classes.

        The class check is the one that matters: a head whose classes were reordered
        still produces logits of the right shape, and every prediction from it is wrong
        in a way nothing else would surface.
        """
        if not self.supports(head):
            raise KeyError(
                f"{head!r} is not a supported head of this checkpoint - treat it as "
                f"unsupported, not as a prediction of none"
            )
        payload = json.dumps(classes, separators=(",", ":")).encode("utf-8")
        actual = hashlib.sha256(payload).hexdigest()[:16]
        expected = self.head_vocabulary_hashes[head]
        if actual != expected:
            raise ValueError(
                f"{head!r} class list does not match the checkpoint "
                f"(expected {expected}, got {actual}) - its logits would be misread"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "modelRevision": self.model_revision,
            "trainingRevision": self.training_revision,
            "supportedHeads": list(self.supported_heads),
            "headVocabularyHashes": dict(self.head_vocabulary_hashes),
            "maxBeamLevels": self.max_beam_levels,
            "maxSlurSlots": self.max_slur_slots,
            "structuredHeadsAutoregressive": self.structured_heads_autoregressive,
            "maxImageHeight": self.max_image_height,
            "maxImageWidth": self.max_image_width,
            "maxSequenceLength": self.max_sequence_length,
            "labelSchemaVersion": self.label_schema_version,
            "datasetRevision": self.dataset_revision,
            "runId": self.run_id,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CapabilityManifest":
        schema = data.get("schemaVersion")
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported capability manifest schema {schema!r}")
        return CapabilityManifest(
            model_revision=data["modelRevision"],
            training_revision=data["trainingRevision"],
            supported_heads=tuple(data["supportedHeads"]),
            head_vocabulary_hashes=dict(data["headVocabularyHashes"]),
            max_beam_levels=data["maxBeamLevels"],
            max_slur_slots=data["maxSlurSlots"],
            structured_heads_autoregressive=data["structuredHeadsAutoregressive"],
            max_image_height=data["maxImageHeight"],
            max_image_width=data["maxImageWidth"],
            max_sequence_length=data["maxSequenceLength"],
            label_schema_version=data["labelSchemaVersion"],
            dataset_revision=data.get("datasetRevision", ""),
            run_id=data.get("runId", ""),
            notes=data.get("notes", {}),
        )


def build(
    *,
    config: Any,
    trained_heads: tuple[str, ...],
    available_heads: tuple[str, ...],
    model_revision: str,
    training_revision: str,
    label_schema_version: str,
    dataset_revision: str = "",
    run_id: str = "",
    notes: dict[str, Any] | None = None,
) -> CapabilityManifest:
    """Declare exactly the heads a run trained.

    `available_heads` is what the architecture built; `trained_heads` is what the run
    actually optimised. Declaring the first would advertise projections that emit
    confident nonsense, which is the failure this manifest exists to prevent - so a
    trained head the architecture does not have is an error, and an untrained head that
    it does have is simply left out.
    """
    unknown = sorted(set(trained_heads) - set(available_heads))
    if unknown:
        raise ValueError(
            f"trained head(s) the architecture does not provide: {unknown} - "
            "the run and the model disagree"
        )
    supported = tuple(head for head in available_heads if head in set(trained_heads))
    return CapabilityManifest(
        model_revision=model_revision,
        training_revision=training_revision,
        supported_heads=supported,
        head_vocabulary_hashes={head: vocabulary_hash(head) for head in supported},
        max_beam_levels=MAX_BEAM_LEVELS,
        max_slur_slots=MAX_SLUR_SLOTS,
        structured_heads_autoregressive=STRUCTURED_HEADS_ARE_AUTOREGRESSIVE,
        max_image_height=getattr(config, "max_height", 0),
        max_image_width=getattr(config, "max_width", 0),
        max_sequence_length=getattr(config, "max_seq_len", 0),
        label_schema_version=label_schema_version,
        dataset_revision=dataset_revision,
        run_id=run_id,
        notes=notes or {},
    )
