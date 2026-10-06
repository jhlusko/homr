import hashlib

import pytest
from PIL import Image, PngImagePlugin

from training.onnx.audit_ossq_pixel_differences import compare_pixels


def test_distinguishes_png_encoding_from_changed_crop_pixels(tmp_path):
    old, new = tmp_path / "old.png", tmp_path / "new.png"
    image = Image.new("RGB", (10, 5), "white")
    image.save(old)
    info = PngImagePlugin.PngInfo()
    info.add_text("encoding", "different")
    image.save(new, pnginfo=info)
    row = dict(
        label="test",
        pixels=str(old),
        public_pixels=str(new),
        pixels_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        public_pixels_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
    )
    assert row["pixels_sha256"] != row["public_pixels_sha256"]
    assert compare_pixels(row)["decoded_pixels_equal"]
    image.putpixel((2, 2), (0, 0, 0))
    image.save(new)
    with pytest.raises(ValueError, match="pixels changed"):
        compare_pixels(row)
    row["public_pixels_sha256"] = hashlib.sha256(new.read_bytes()).hexdigest()
    result = compare_pixels(row)
    assert not result["decoded_pixels_equal"]
    assert result["different_pixels"] == 1
