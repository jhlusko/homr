import unittest

from homr.transformer.capability_manifest import (
    SCHEMA_VERSION,
    CapabilityManifest,
    build,
    vocabulary_hash,
)
from homr.transformer.structured_notation import BEAM_LEVEL_CLASSES, STEM_CLASSES
from training.architecture.transformer.structured_heads import head_names


class _Config:
    max_height = 256
    max_width = 1280
    max_seq_len = 608


def _manifest(trained: tuple[str, ...] | None = None) -> CapabilityManifest:
    available = tuple(head_names(beam_levels=2, slur_slots=1))
    return build(
        config=_Config(),
        trained_heads=trained if trained is not None else available,
        available_heads=available,
        model_revision="abc123",
        training_revision="def456",
        label_schema_version="homr.structured-symbols.v1",
    )


class TestDeclaringOnlyWhatWasTrained(unittest.TestCase):
    def test_an_untrained_head_is_not_declared(self) -> None:
        # The projections exist and emit logits from the moment the architecture changes.
        # Declaring one that was never optimised advertises confident nonsense.
        manifest = _manifest(trained=("beam.level.1", "stem.direction"))

        self.assertEqual(manifest.supported_heads, ("beam.level.1", "stem.direction"))
        self.assertFalse(manifest.supports("beam.level.2"))
        self.assertFalse(manifest.supports("slur.slot.1.event"))

    def test_a_trained_head_the_architecture_lacks_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build(
                config=_Config(),
                trained_heads=("beam.level.5",),
                available_heads=tuple(head_names(beam_levels=2, slur_slots=1)),
                model_revision="a",
                training_revision="b",
                label_schema_version="v1",
            )

        self.assertIn("beam.level.5", str(ctx.exception))

    def test_declared_order_follows_the_architecture_not_the_run(self) -> None:
        manifest = _manifest(trained=("stem.direction", "beam.level.1"))

        self.assertEqual(manifest.supported_heads, ("beam.level.1", "stem.direction"))


class TestUnsupportedHeads(unittest.TestCase):
    def test_asking_for_an_undeclared_head_raises_rather_than_returning_none(self) -> None:
        manifest = _manifest(trained=("stem.direction",))

        with self.assertRaises(KeyError) as ctx:
            manifest.check_compatible("beam.level.1", tuple(str(s) for s in BEAM_LEVEL_CLASSES))

        self.assertIn("not as a prediction of none", str(ctx.exception))


class TestClassOrder(unittest.TestCase):
    def test_matching_classes_pass(self) -> None:
        manifest = _manifest()

        manifest.check_compatible("stem.direction", tuple(str(s) for s in STEM_CLASSES))

    def test_a_reordered_class_list_is_caught(self) -> None:
        # Logits are positional: index 3 of a beam head means BACKWARD_HOOK because that
        # is where it sits. Reordering silently reinterprets every prediction, with no
        # shape error anywhere to notice it.
        manifest = _manifest()
        reordered = tuple(reversed([str(state) for state in STEM_CLASSES]))

        with self.assertRaises(ValueError) as ctx:
            manifest.check_compatible("stem.direction", reordered)

        self.assertIn("would be misread", str(ctx.exception))

    def test_hashes_differ_between_heads_with_different_classes(self) -> None:
        self.assertNotEqual(vocabulary_hash("beam.level.1"), vocabulary_hash("stem.direction"))

    def test_hashes_match_between_heads_sharing_a_class_list(self) -> None:
        self.assertEqual(vocabulary_hash("beam.level.1"), vocabulary_hash("beam.level.2"))

    def test_an_unknown_head_has_no_class_list(self) -> None:
        with self.assertRaises(KeyError):
            vocabulary_hash("lyrics.syllabic")


class TestRoundTrip(unittest.TestCase):
    def test_survives_serialisation(self) -> None:
        original = _manifest()

        restored = CapabilityManifest.from_dict(original.to_dict())

        self.assertEqual(restored, original)

    def test_records_the_limits_and_feedback_mode(self) -> None:
        manifest = _manifest()

        self.assertEqual(manifest.max_sequence_length, 608)
        self.assertEqual(manifest.max_image_width, 1280)
        # Output-only in this phase; a consumer cannot tell from the weights.
        self.assertFalse(manifest.structured_heads_autoregressive)

    def test_an_unknown_schema_is_refused(self) -> None:
        data = _manifest().to_dict()
        data["schemaVersion"] = "homr.capability-manifest.v99"

        with self.assertRaises(ValueError):
            CapabilityManifest.from_dict(data)

    def test_schema_version_is_stamped(self) -> None:
        self.assertEqual(_manifest().to_dict()["schemaVersion"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
