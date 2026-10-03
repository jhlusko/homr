"""Isolated matched float32 export of accepted September core and v7 heads."""
import argparse
import json
import os
from pathlib import Path
import time

from training.onnx.qualify_joint_export import config, setup, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['HOMR_MODEL_NAME'] = 'accepted-core-20260918-v7-heads-20260924-qual20261003'
    setup()
    import torch
    import onnx
    from training.onnx import convert
    from training.onnx.split_pinned_checkpoint import split_checkpoint
    spec = json.loads((args.source_run / 'models.json').read_text())['accepted_oldcore_v7']
    for key, digest in spec['sha256'].items():
        if sha(spec[key]) != digest:
            raise ValueError('Accepted checkpoint changed')
    working = out / 'working-weights'
    working.mkdir(exist_ok=True)
    encoder, decoder = split_checkpoint(torch.load(spec['core'], map_location='cpu', weights_only=True))
    torch.save(encoder, working / 'encoder_weights.pt')
    torch.save(decoder, working / 'decoder_weights.pt')
    convert.Config = config
    previous = Path.cwd()
    artifacts = {}
    try:
        os.chdir(working)
        for component, function in (
            ('encoder', lambda: convert.convert_encoder(True, str(out))),
            ('decoder', lambda: convert.convert_decoder(True, str(out))),
            ('heads', lambda: convert.convert_structured_heads(True, spec['heads'], str(out))),
        ):
            tick = time.monotonic()
            path = Path(function())
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            artifacts[component] = dict(path=str(path), sha256=sha(path),
                                       seconds=time.monotonic()-tick, bytes=path.stat().st_size)
            print('Exported accepted', component, flush=True)
    finally:
        os.chdir(previous)
    write(out / 'exports.json', artifacts)
    write(out / 'identity.json', dict(model=spec, threads=1, torch=str(torch.__version__),
        script_sha256=sha(__file__),
        scope='Matched float32 corrected-context export; accepted weight pairing, not deployed quantized/provider pipeline'))


if __name__ == '__main__':
    main()
