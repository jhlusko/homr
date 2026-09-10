import unittest

from training.omr_datasets.split_pairs_by_score import (
    RARE_TOPOLOGIES,
    rare_topologies_of_score,
    score_of,
    split_by_score,
)


def lines_for(scores: dict[str, int]) -> list[str]:
    out = []
    for score_id, n in scores.items():
        for i in range(n):
            out.append(f"/p/{score_id}-sys{i}-v0.png,/p/{score_id}-sys{i}-v0.tokens")
    return out


class TestRareTopologiesOfScore(unittest.TestCase):
    """Derived from the manifest, not from the alignment's status field: a system
    can be `aligned` and still emit no pair, and that proxy matched 54 scores where
    18 hold every rare pair."""

    TOPO = {"A": {0: "one-to-one", 1: "many-to-many"}, "B": {0: "reference-line-split"}}

    def test_only_scores_present_in_the_manifest_count(self) -> None:
        lines = ["/p/A-sys0-v0.png,/p/A-sys0-v0.tokens"]
        self.assertEqual(rare_topologies_of_score(lines, self.TOPO), {})

    def test_each_scores_own_rare_kinds_are_reported(self) -> None:
        lines = ["/p/A-sys1-v0.png,/p/A-sys1-v0.tokens",
                 "/p/B-sys0-v0.png,/p/B-sys0-v0.tokens"]
        self.assertEqual(
            rare_topologies_of_score(lines, self.TOPO),
            {"A": frozenset({"many-to-many"}), "B": frozenset({"reference-line-split"})},
        )

    def test_rare_topologies_are_the_non_one_to_one_ones(self) -> None:
        self.assertNotIn("one-to-one", RARE_TOPOLOGIES)
        self.assertIn("many-to-many", RARE_TOPOLOGIES)


class TestSplit(unittest.TestCase):
    def test_split_is_score_disjoint(self) -> None:
        lines = lines_for({f"S{i}": 3 for i in range(40)})
        train, val = split_by_score(lines, 0.2)
        self.assertEqual({score_of(x) for x in train} & {score_of(x) for x in val}, set())

    def test_unstratified_splitting_is_blind_to_topology(self) -> None:
        """The 2026-08-27 failure, reproduced: with no stratification the rare
        scores follow the hash, and here every one of them lands in train, leaving
        validation unable to exercise a single non-one-to-one system."""
        lines = lines_for({f"S{i}": 3 for i in range(40)})
        _, val = split_by_score(lines, 0.1)
        val_scores = {score_of(x) for x in val}
        rare = {s: frozenset({"many-to-many"})
                for s in (f"S{i}" for i in range(40)) if s not in val_scores}
        self.assertTrue(rare, "expected some scores outside validation")
        # Those same scores DO reach validation once the split is told about them.
        _, val2 = split_by_score(lines, 0.1, rare_by_score=rare)
        self.assertTrue(set(rare) & {score_of(x) for x in val2})

    def test_stratifying_puts_rare_topology_scores_in_validation(self) -> None:
        rare = {"S3": frozenset({"many-to-many"}), "S7": frozenset({"many-to-many"})}
        lines = lines_for({f"S{i}": 3 for i in range(40)})
        _, val = split_by_score(lines, 0.1, rare_by_score=rare)
        self.assertTrue(set(rare) & {score_of(x) for x in val},
                        "a rare-topology score must reach validation")

    def test_a_small_rare_stratum_still_yields_a_validation_score(self) -> None:
        """18 rare scores at 10% can hash to none; the guarantee must not depend
        on the hash being kind."""
        rare = {"R1": frozenset({"many-to-many"})}
        lines = lines_for({"R1": 3, **{f"S{i}": 3 for i in range(40)}})
        _, val = split_by_score(lines, 0.0001, rare_by_score=rare)
        self.assertIn("R1", {score_of(x) for x in val})

    def test_stratified_split_is_still_score_disjoint(self) -> None:
        rare = {s: frozenset({"many-to-many"}) for s in ("S3", "S7", "S11")}
        lines = lines_for({f"S{i}": 3 for i in range(40)})
        train, val = split_by_score(lines, 0.15, rare_by_score=rare)
        self.assertEqual({score_of(x) for x in train} & {score_of(x) for x in val}, set())

    def test_split_is_deterministic(self) -> None:
        rare = {"S3": frozenset({"many-to-many"})}
        lines = lines_for({f"S{i}": 2 for i in range(30)})
        self.assertEqual(split_by_score(lines, 0.1, rare), split_by_score(lines, 0.1, rare))


if __name__ == "__main__":
    unittest.main()


class TestEachRareKindReachesValidation(unittest.TestCase):
    def test_both_kinds_get_their_own_stratum(self) -> None:
        """A single rare/not-rare split gave validation four reference-line-split
        pairs and zero many-to-many.  Each kind needs its own stratum."""
        rare = {"M1": frozenset({"many-to-many"}),
                "R1": frozenset({"reference-line-split"})}
        lines = lines_for({"M1": 3, "R1": 3, **{f"S{i}": 3 for i in range(40)}})
        _, val = split_by_score(lines, 0.0001, rare_by_score=rare)
        val_scores = {score_of(x) for x in val}
        self.assertIn("M1", val_scores)
        self.assertIn("R1", val_scores)


if __name__ == "__main__":
    unittest.main()
