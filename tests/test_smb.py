"""SMB uses a page transcription as its benchmark reference."""

from validation.smb import _sample_kern


def test_page_kern_precedes_regions() -> None:
    assert (
        _sample_kern(
            {
                "page": {"kern": "**kern\n4c\n*-"},
                "regions": [{"kern": "**kern\n4d\n*-"}],
            }
        )
        == "**kern\n4c\n*-"
    )


def test_old_region_only_shape_remains_supported() -> None:
    assert _sample_kern({"regions": [{"kern": "a"}, {"kern": ""}, {"kern": "b"}]}) == "a\nb"


def test_top_level_kern_precedes_old_region_fallback() -> None:
    assert _sample_kern({"kern": "page", "regions": [{"kern": "region"}]}) == "page"
