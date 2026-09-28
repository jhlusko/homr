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


def test_hook_examples_have_isolated_deepest_beams_and_complete_measures() -> None:
    for family in (7, 11, 15):
        for variant in (2, 3):
            xml, info = build_score(20260928, family, variant)
            depth = info['beam_depth']
            for measure in ET.fromstring(xml).findall('./part/measure'):
                assert sum(int(n.findtext('duration')) for n in measure.findall('note')) == 960
                notes = [n for n in measure.findall('note') if n.find('rest') is None]
                assert len(notes) == 2
                beams = [n.find(f"beam[@number='{depth}']") for n in notes]
                assert sum(b is not None for b in beams) == 1
                assert beams[0 if variant == 2 else 1].text == ('forward hook' if variant == 2 else 'backward hook')


def test_renderer_audit_rejects_changed_beam_labels() -> None:
    from training.omr_datasets.synthetic_head_curriculum import audit_renderer_labels
    original, _ = build_score(20260928, 7, 2)
    assert not audit_renderer_labels(original, original)['differences']
    changed = original.replace('forward hook', 'backward hook')
    assert any(d['field'] == 'beams' for d in audit_renderer_labels(original, changed)['differences'])


def test_reversed_overlap_changes_sides_only_and_preserves_v4_identity():
    from training.omr_datasets.synthetic_head_curriculum import parse_symbols
    from training.transformer.training_vocabulary import token_lines_to_str
    from homr.transformer.structured_notation import SlurSide, SlurEvent
    for family in (2, 6, 10, 14):
        original, old = build_score(20260928, family, 3)
        reversed_xml, new = build_score(20260928, family, 4)
        assert old['seed'] == new['seed'] and old['split'] == new['split']
        assert token_lines_to_str(parse_symbols(original)) == token_lines_to_str(parse_symbols(reversed_xml))
        notes, findings = parse_score(ET.fromstring(reversed_xml))
        assert findings.clean
        second = [n.slurs[1] for part in notes.values() for n in part if n.slurs[1][0] != SlurEvent.NONE]
        assert second and all(side == SlurSide.ABOVE for _, side in second)


def test_supplement_keeps_published_v4_family_seed_and_split():
    from training.omr_datasets.synthetic_head_curriculum import family_seed
    assert family_seed(20260928, 3) == 4207139528
    assert split_for(20260928, 11) == 'test'
    assert split_for(20260928, 15) == 'valid'
