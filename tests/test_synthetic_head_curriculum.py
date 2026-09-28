import xml.etree.ElementTree as ET

from training.omr_datasets.structured_notation_parser import parse_score
from training.omr_datasets.synthetic_head_curriculum import (
    KINDS,
    build_score,
    split_for,
)


def test_each_family_variant_has_auditable_structured_labels() -> None:
    seed = 20260928
    for family, kind in enumerate(KINDS):
        for variant in range(4):
            xml, metadata = build_score(seed, family, variant)
            assert metadata["kind"] == kind
            assert metadata["variant"] == variant

            _, findings = parse_score(ET.fromstring(xml))
            assert findings.clean
            assert findings.ambiguous_beaming == 0


def test_all_variants_of_a_motif_stay_in_one_deterministic_split() -> None:
    seed = 20260928
    for family in range(256):
        expected = split_for(seed, family)
        assert expected in {"train", "valid", "test"}
        assert all(build_score(seed, family, variant)[1]["split"] == expected
                   for variant in range(4))
        assert split_for(seed, family) == expected
