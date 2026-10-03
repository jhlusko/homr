"""Make legacy-exported cross-attention head splits safe for empty context."""
import onnx
import numpy as np
from onnx import helper, numpy_helper

CAPABILITY = 'homr.decoder.empty_continuation_context'


def enable_empty_context(graph, heads: int, depth: int) -> list[str]:
    """Fail closed if the expected legacy exporter layout has changed."""
    if any(p.key == CAPABILITY for p in graph.metadata_props):
        raise ValueError('Decoder context capability already present')
    matches = [n for n in graph.graph.node if n.op_type == 'Reshape'
               and ('split_k_heads' in n.name or 'split_v_heads' in n.name)
               and 'Concat' in n.input[1]]
    if heads <= 0 or len(matches) != 2 * depth:
        raise ValueError(f'Unexpected cross-attention head split layout: {len(matches)}')
    names = {n.name for n in matches}
    nodes = []
    for node in graph.graph.node:
        if node.name in names:
            prefix = node.name + '/empty_context'
            minus, count = prefix + '/minus', prefix + '/heads'
            graph.graph.initializer.extend([
                numpy_helper.from_array(np.array(-1, dtype=np.int64), minus),
                numpy_helper.from_array(np.array(heads, dtype=np.int64), count)])
            mask, shape = prefix + '/mask', prefix + '/shape'
            nodes.extend([helper.make_node('Equal', [node.input[1], minus], [mask]),
                          helper.make_node('Where', [mask, count, node.input[1]], [shape])])
            node.input[1] = shape
            del node.attribute[:]
            node.attribute.append(helper.make_attribute('allowzero', 1))
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    graph.metadata_props.add(key=CAPABILITY, value='1')
    onnx.checker.check_model(graph)
    return [n.name for n in matches]
