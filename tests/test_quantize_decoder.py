"""`quantize_decoder` reproduces the production decoder's storage format.

The pinned decoder ONNX is ~4x smaller than a plain fp32 export of the same weights
because it went through ONNX Runtime's dynamic int8 quantization after export - confirmed
by comparing initializer byte counts per shape and by the real graph's node types
(`DynamicQuantizeLinear`, `MatMulInteger`). This wraps `onnxruntime.quantization.
quantize_dynamic` so that step has a place in this project's own export tooling.

These tests use a tiny synthetic graph rather than a real decoder export: the point is
the wrapper's contract (the overwrite guard, that quantization actually shrinks a graph
built from real MatMul weights), not re-deriving the accuracy result already recorded in
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` - that check needs the real trained weights and
real staff data, not a fixture.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from training.onnx.convert import quantize_decoder

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False


def _matmul_graph(path: Path) -> None:
    """A graph with one large MatMul weight - big enough to be worth quantizing."""
    weight = np.random.randn(256, 256).astype(np.float32)
    weight_init = helper.make_tensor(
        "weight", TensorProto.FLOAT, weight.shape, weight.flatten().tolist()
    )
    node = helper.make_node("MatMul", ["input", "weight"], ["output"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 256])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256])],
        [weight_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


@unittest.skipUnless(_HAS_ONNX, "onnx/onnxruntime not installed")
class TestQuantizeDecoder(unittest.TestCase):
    def test_it_shrinks_a_graph_with_a_large_matmul_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.onnx"
            dst = Path(directory) / "dst.onnx"
            _matmul_graph(src)

            result = quantize_decoder(str(src), str(dst))

            self.assertEqual(result, str(dst))
            self.assertTrue(dst.exists())
            self.assertLess(dst.stat().st_size, src.stat().st_size)

    def test_it_refuses_to_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.onnx"
            dst = Path(directory) / "dst.onnx"
            _matmul_graph(src)
            dst.write_bytes(b"already here")

            result = quantize_decoder(str(src), str(dst))

            self.assertIsNone(result)
            self.assertEqual(dst.read_bytes(), b"already here")

    def test_overwrite_true_replaces_the_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "src.onnx"
            dst = Path(directory) / "dst.onnx"
            _matmul_graph(src)
            dst.write_bytes(b"stale")

            result = quantize_decoder(str(src), str(dst), overwrite=True)

            self.assertEqual(result, str(dst))
            self.assertNotEqual(dst.read_bytes(), b"stale")


if __name__ == "__main__":
    unittest.main()
