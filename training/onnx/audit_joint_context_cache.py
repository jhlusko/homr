"""Test empty continuation context without changing production decoder behavior."""
import argparse
import json
from pathlib import Path

from training.onnx.qualify_joint_export import setup, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--export-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repair-empty-reshape', action='store_true')
    parser.add_argument('--full-parity', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from training.onnx.qualify_joint_export import sha
    write(args.out / 'diagnostic-identity.json', {
        'script_sha256': sha(__file__), 'source_run': str(args.source_run.resolve()),
        'export_run': str(args.export_run.resolve()), 'threads': 1,
        'empty_continuation_context': True, 'repair_empty_reshape': args.repair_empty_reshape,
        'full_parity': args.full_parity,
        'scope': 'Isolated diagnostic graph and runtime monkeypatch; shipping runtime unchanged'})
    setup()
    from homr.transformer.decoder_inference import ScoreDecoder
    from training.onnx.audit_joint_onnx_parity import isolate
    rows = [json.loads(line) for line in (args.export_run / 'parity.jsonl').open()]
    row = next(r for r in rows if not r['recognition_equal'])
    original = ScoreDecoder.generate
    traces = []

    def generate(self, *positional, **kwargs):
        context = kwargs['context']
        run = self.net.run_with_iobinding
        steps = []

        def corrected(*a, **kw):
            step = len(steps)
            if step:
                self.io_binding.bind_cpu_input('context', context[:, :0])
            result = run(*a, **kw)
            outputs = self.io_binding.get_outputs()
            steps.append({'step': step, 'context_tokens': context.shape[1] if not step else 0,
                          'cache_shapes': [v.shape() for v in outputs[8:]]})
            return result

        self.net.run_with_iobinding = corrected
        try:
            return original(self, *positional, **kwargs)
        finally:
            self.net.run_with_iobinding = run
            traces.append(steps)
            write(args.out / 'cache-traces.json', traces)

    ScoreDecoder.generate = generate
    # isolate reads graph identities from its output directory; retain originals.
    exports = json.loads((args.export_run / 'exports.json').read_text())
    if args.repair_empty_reshape:
        import onnx
        import numpy as np
        from onnx import helper, numpy_helper
        from training.onnx.qualify_joint_export import config, sha
        graph = onnx.load(exports['decoder']['path'])
        nodes = []
        repaired = []
        for node in graph.graph.node:
            # Only cross-attention head splits use a dynamic context length.
            if node.op_type == 'Reshape' and ('split_k_heads' in node.name or 'split_v_heads' in node.name) and 'Concat' in node.input[1]:
                prefix = node.name + '/empty_context'
                minus, heads = prefix + '/minus', prefix + '/heads'
                graph.graph.initializer.extend([
                    numpy_helper.from_array(np.array(-1, dtype=np.int64), minus),
                    numpy_helper.from_array(np.array(config().decoder_heads, dtype=np.int64), heads)])
                mask, shape = prefix + '/mask', prefix + '/shape'
                nodes.extend([helper.make_node('Equal', [node.input[1], minus], [mask]),
                              helper.make_node('Where', [mask, heads, node.input[1]], [shape])])
                node.input[1] = shape
                del node.attribute[:]
                node.attribute.append(helper.make_attribute('allowzero', 1))
                repaired.append(node.name)
            nodes.append(node)
        if len(repaired) != 16:
            raise ValueError(f'Unexpected cross-attention split count: {len(repaired)}')
        del graph.graph.node[:]
        graph.graph.node.extend(nodes)
        onnx.checker.check_model(graph)
        path = args.out / 'decoder-empty-context.onnx'
        onnx.save(graph, path)
        exports['decoder'] = dict(exports['decoder'], path=str(path.resolve()), sha256=sha(path))
        write(args.out / 'reshape-repair.json', {'nodes': repaired, 'heads': config().decoder_heads,
              'original_decoder': json.loads((args.export_run / 'exports.json').read_text())['decoder']})
    write(args.out / 'exports.json', exports)
    try:
        if args.full_parity:
            from training.onnx.qualify_joint_export import parity
            parity(args.source_run, args.out)
        else:
            isolate(args.source_run, args.out, row)
    except Exception as error:
        write(args.out / 'ERROR.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
