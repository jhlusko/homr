"""Check actual greedy, margin and forced-prefix runtime paths on a fixed crop."""
import argparse
import json
from pathlib import Path

import numpy as np

from training.onnx.qualify_joint_export import config, setup, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--original-run', type=Path, required=True)
    args = parser.parse_args()
    setup()
    from homr.transformer.staff2score import Staff2Score
    from homr.transformer.vocabulary import EncodedSymbol
    from training.transformer.qualify_joint_holdout import pixels, serialize
    row = next(r for r in map(json.loads, (args.original_run / 'parity.jsonl').open())
               if not r['recognition_equal'])
    exports = json.loads((args.run / 'exports.json').read_text())
    cfg = config()
    for component, attr in [('encoder', 'encoder_path'), ('decoder', 'decoder_path'),
                            ('heads', 'structured_heads_path')]:
        if sha(exports[component]['path']) != exports[component]['sha256']:
            raise ValueError('Graph changed')
        setattr(cfg.filepaths, attr, exports[component]['path'])
    model = Staff2Score(cfg)
    context = model.encoder.generate(pixels(row['image']).unsqueeze(0).numpy())
    if not model.decoder.empty_continuation_context:
        raise ValueError('Missing graph capability')
    traces = []
    run = model.decoder.net.run_with_iobinding

    def capture(*a, **kw):
        result = run(*a, **kw)
        shapes = [v.shape() for v in model.decoder.io_binding.get_outputs()[8:]]
        # Two self-attention and two cross-attention tensors per decoder layer.
        cross_lengths = [s[-2] for i, s in enumerate(shapes) if i % 4 >= 2]
        if any(length != context.shape[1] for length in cross_lengths):
            raise ValueError('Cross-attention cache grew during decoding')
        traces.append(cross_lengths)
        return result

    model.decoder.net.run_with_iobinding = capture
    start, nonote = np.array([[1]], dtype=np.int64), np.array([[0]], dtype=np.int64)
    kwargs = dict(context=context, eos_token=cfg.eos_token, seq_len=cfg.max_seq_len)
    greedy = model.decoder.generate(start, nonote, **kwargs)
    margins, _, _ = model.decoder.generate_with_rhythm_margins(start, nonote, **kwargs)
    prefix_length = min(20, len(greedy) // 2)
    continuation = model.decoder.generate_from_prefix(
        [EncodedSymbol('BOS'), *greedy[:prefix_length]], **kwargs)
    fork_step = next(i for i, symbol in enumerate(greedy) if symbol.rhythm.startswith('note_'))
    alternative = 'note_4' if greedy[fork_step].rhythm != 'note_4' else 'note_8'
    forked = model.decoder.rhythm_alternative(
        greedy, fork_step, cfg.rhythm_vocab[alternative], **kwargs)
    expected = [serialize(s) for s in greedy]
    report = dict(id=row['id'], graphs=exports, script_sha256=sha(__file__), threads=1,
                  capability=True, greedy_matches_original_eager=expected == row['eager'],
                  margins_match_greedy=[serialize(s) for s in margins] == expected,
                  prefix_matches_greedy=[serialize(s) for s in continuation] == expected[prefix_length:],
                  fork_prefix_preserved=[serialize(s) for s in forked[:fork_step]] == expected[:fork_step],
                  fork_rhythm_forced=forked[fork_step].rhythm == alternative,
                  fork_step=fork_step, fork_alternative=alternative,
                  prefix_length=prefix_length, steps=len(traces), cross_cache_tokens=context.shape[1])
    write(args.run / 'decoder-paths.json', report)
    write(args.run / 'page-models.json', {'candidate': exports})
    if not all(report[k] for k in ('greedy_matches_original_eager', 'margins_match_greedy',
                                 'prefix_matches_greedy', 'fork_prefix_preserved', 'fork_rhythm_forced')):
        raise ValueError('Runtime decoder path parity failed')
    print('Greedy, margin, prefix and altered-rhythm fork paths passed; cross-attention cache constant.', flush=True)


if __name__ == '__main__':
    main()
