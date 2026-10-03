"""Exercise zero-length ONNX reshapes and legacy capability selection."""
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from homr.transformer.configs import Config
from homr.transformer.decoder_inference import ScoreDecoder
from training.onnx.decoder_context import enable_empty_context


def graph():
    constants = [numpy_helper.from_array(np.array([0, 1], dtype=np.int64), 'indices'),
                 numpy_helper.from_array(np.array([-1, 64], dtype=np.int64), 'tail')]
    nodes = [helper.make_node('Shape', ['input'], ['size']),
             helper.make_node('Gather', ['size', 'indices'], ['leading']),
             helper.make_node('Concat', ['leading', 'tail'], ['Concat_shape'], axis=0)]
    for kind in ('k', 'v'):
        nodes.append(helper.make_node('Reshape', ['input', 'Concat_shape'], [kind],
                     name=f'/cross/split_{kind}_heads/Reshape'))
    model = helper.make_model(helper.make_graph(nodes, 'test',
        [helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, None, 512])],
        [helper.make_tensor_value_info(k, onnx.TensorProto.FLOAT, [1, None, 8, 64])
         for k in ('k', 'v')], initializer=constants), opset_imports=[helper.make_opsetid('', 18)])
    model.ir_version = 9
    return model


class TestDecoderEmptyContext(unittest.TestCase):
    def session(self, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                    providers=['CPUExecutionProvider'])

    def test_empty_and_nonempty_context_have_exact_head_shapes(self):
        model = graph()
        original = self.session(model)
        values = np.arange(1024, dtype=np.float32).reshape(1, 2, 512)
        expected = original.run(None, {'input': values})
        enable_empty_context(model, heads=8, depth=1)
        corrected = self.session(model)
        for a, b in zip(expected, corrected.run(None, {'input': values}), strict=True):
            np.testing.assert_array_equal(a, b)
        for result in corrected.run(None, {'input': values[:, :0]}):
            self.assertEqual(result.shape, (1, 0, 8, 64))
        context = np.zeros((1, 20, 512), dtype=np.float32)
        self.assertEqual(ScoreDecoder(original, False, False, Config())._continuation_context(context).shape[1], 1)
        self.assertEqual(ScoreDecoder(corrected, False, False, Config())._continuation_context(context).shape[1], 0)

    def test_unknown_export_layout_does_not_receive_capability(self):
        model = graph()
        with self.assertRaises(ValueError):
            enable_empty_context(model, heads=8, depth=2)
        self.assertFalse(model.metadata_props)
