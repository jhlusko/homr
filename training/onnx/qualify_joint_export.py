"""Isolated joint-checkpoint export and fixed-crop eager/ONNX parity qualification.

Never writes the shipping graph cache or promotes a model. All CPU pools use one
thread. Float32 export parity does not establish quantized/GPU/full-page parity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def setup():
    import torch
    import cv2
    import onnxruntime as ort
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(20261002)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    original = ort.InferenceSession

    class SingleThreadSession(original):
        def __init__(self, *args, **kwargs):
            options = kwargs.get('sess_options') or ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            kwargs['sess_options'] = options
            super().__init__(*args, **kwargs)

    ort.InferenceSession = SingleThreadSession


def config():
    from homr.transformer.configs import Config
    cfg = Config()
    cfg.encoder_pretrained = False
    cfg.enable_structured_heads = True
    cfg.structured_beam_levels, cfg.structured_slur_slots = 4, 2
    cfg.enforce_final_divider = False
    cfg.use_gpu_inference = cfg.use_coreml_encoder = False
    return cfg


def provenance(source, out):
    import torch
    import onnxruntime
    spec = json.loads((source / 'models.json').read_text())['public_joint27']
    if sha(spec['joint']) != spec['sha256']['joint']:
        raise ValueError('Joint checkpoint changed')
    root = Path(__file__).resolve().parents[2]
    files = ['training/onnx/convert.py', 'training/onnx/decoder_context.py', 'training/onnx/qualify_joint_export.py',
             'homr/transformer/configs.py', 'homr/transformer/decoder_inference.py',
             'homr/transformer/encoder_inference.py', 'homr/transformer/structured_decode.py',
             'training/architecture/transformer/encoder.py',
             'training/architecture/transformer/decoder.py',
             'training/architecture/transformer/custom_x_transformer.py']
    identity = dict(model=spec, torch=str(torch.__version__), onnxruntime=onnxruntime.__version__,
                    code_sha256={p: sha(root / p) for p in files}, threads=1,
                    precision='float32; legacy TorchScript exporter; no quantization',
                    policy='raw greedy crop outputs; final-divider enforcement off; no v7 confidence calibration')
    path = out / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Qualification identity changed; use a new output directory')
    write(path, identity)
    return spec


def export(source, out):
    import torch
    import onnx
    from training.onnx import convert
    from training.onnx.split_pinned_checkpoint import split_checkpoint
    spec = provenance(source, out)
    if (out / 'exports.json').exists():
        for record in json.loads((out / 'exports.json').read_text()).values():
            if sha(record['path']) != record['sha256']:
                raise ValueError('Export artifact changed')
        return
    state = torch.load(spec['joint'], map_location='cpu', weights_only=True)
    encoder, decoder = split_checkpoint(state)
    heads = {key.split('decoder.structured_heads.', 1)[1]: value
             for key, value in state.items() if key.startswith('decoder.structured_heads.')}
    if not encoder or not decoder or not heads:
        raise ValueError('Incomplete joint checkpoint')
    weights = out / 'working-weights'
    weights.mkdir(exist_ok=True)
    torch.save(encoder, weights / 'encoder_weights.pt')
    torch.save(decoder, weights / 'decoder_weights.pt')
    torch.save(heads, weights / 'heads.pt')
    convert.Config = config  # configure existing strict exporters without editing their defaults
    previous = Path.cwd()
    artifacts = {}
    try:
        os.chdir(weights)
        for name, function in (
            ('encoder', lambda: convert.convert_encoder(True, str(out))),
            ('decoder', lambda: convert.convert_decoder(True, str(out))),
            ('heads', lambda: convert.convert_structured_heads(True, str(weights / 'heads.pt'), str(out))),
        ):
            write(out / 'STATUS.json', dict(stage='exporting', component=name, updated=time.time()))
            tick = time.monotonic()
            path = Path(function())
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
                raise ValueError('Export unexpectedly depends on external weights')
            artifacts[name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                                   seconds=time.monotonic()-tick,
                                   inputs=[v.name for v in graph.graph.input], outputs=[v.name for v in graph.graph.output])
            print('Exported', name, artifacts[name]['seconds'], 'seconds', flush=True)
        write(out / 'exports.json', artifacts)
    finally:
        os.chdir(previous)


def parity(source, out):
    import torch
    import onnxruntime as ort
    from training.architecture.transformer.tromr_arch import TrOMR
    from homr.transformer.staff2score import Staff2Score
    from training.transformer.qualify_joint_holdout import pixels, serialize
    spec = provenance(source, out)
    artifacts = json.loads((out / 'exports.json').read_text())
    for record in artifacts.values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Export changed')
    cfg = config()
    cfg.filepaths.encoder_path = artifacts['encoder']['path']
    cfg.filepaths.decoder_path = artifacts['decoder']['path']
    cfg.filepaths.structured_heads_path = artifacts['heads']['path']
    onnx_model = Staff2Score(cfg)
    eager = TrOMR(cfg).eval()
    eager.load_state_dict(torch.load(spec['joint'], map_location='cpu', weights_only=True), strict=True)
    eager.decoder.device = torch.device('cpu')
    # Teacher-independent projection check includes whole-sequence and step shapes.
    heads_session = ort.InferenceSession(artifacts['heads']['path'], providers=['CPUExecutionProvider'])
    projection_checks = []
    for length in (1, 3):
        hidden = torch.randn(1, length, cfg.decoder_dim)
        with torch.inference_mode():
            expected = eager.decoder.structured_heads(hidden)
        actual = heads_session.run(None, {'hidden': hidden.numpy()})
        for name, values in zip(artifacts['heads']['outputs'], actual, strict=True):
            reference = expected[name].numpy()
            projection_checks.append(dict(head=name, length=length,
                max_abs=float(np.max(np.abs(reference-values))),
                close=bool(np.allclose(reference, values, rtol=1e-4, atol=1e-5)),
                argmax_equal=bool(np.array_equal(reference.argmax(-1), values.argmax(-1)))))
    rows = [json.loads(line) for line in (source / 'pilot.jsonl').open()]
    cached = {r['id']: r for r in map(json.loads, (source / 'holdout/public_joint27.jsonl').open())}
    results = []
    with (out / 'parity.jsonl').open('w') as handle:
        for index, row in enumerate(rows):
            tick = time.monotonic()
            if sha(row['image']) != row['image_sha256']:
                raise ValueError('Fixed crop changed')
            tensor = pixels(row['image']).unsqueeze(0)
            with torch.inference_mode():
                context = eager.encoder(tensor).numpy()
                eager_symbols = [serialize(s) for s in eager.generate(tensor)]
            encoder_values = onnx_model.encoder.generate(tensor.numpy())
            symbols = onnx_model.decoder.generate(np.array([[1]], dtype=np.int64),
                np.array([[0]], dtype=np.int64), seq_len=cfg.max_seq_len, eos_token=cfg.eos_token,
                context=encoder_values)
            onnx_symbols = [serialize(s) for s in symbols]
            token_fields = ('rhythm', 'pitch', 'lift', 'articulation', 'slur', 'position')
            tokens = lambda values: [{k: r[k] for k in token_fields} for r in values]
            record = dict(**row, seconds=time.monotonic()-tick,
                encoder_max_abs=float(np.max(np.abs(context-encoder_values))),
                encoder_close=bool(np.allclose(context, encoder_values, rtol=1e-3, atol=1e-4)),
                recognition_equal=tokens(eager_symbols)==tokens(onnx_symbols),
                all_notation_equal=eager_symbols==onnx_symbols,
                eager_cached_equal=eager_symbols==cached[row['id']]['predicted'],
                eager=eager_symbols, onnx=onnx_symbols)
            handle.write(json.dumps(record)+'\n'); handle.flush(); results.append(record)
            write(out / 'STATUS.json', dict(stage='parity', completed=index+1, total=len(rows),
                                            last_seconds=record['seconds'], updated=time.time()))
            print(f'Parity {index+1}/{len(rows)} {row["score"]}: recognition={record["recognition_equal"]}, notation={record["all_notation_equal"]}', flush=True)
    gates = ('encoder_close', 'recognition_equal', 'all_notation_equal', 'eager_cached_equal')
    passed = all(all(r[k] for k in gates) for r in results) and all(r['close'] and r['argmax_equal'] for r in projection_checks)
    write(out / 'parity-summary.json', dict(passed=passed, crops=len(results),
        counts={k: sum(r[k] for r in results) for k in gates}, projection_checks=projection_checks,
        limitations='Float32 CPU and identical normalized tensors. Quantized exports, image-transform rounding, full-page/runtime policies and CUDA still require separate checks.'))
    write(out / 'STATUS.json', dict(stage='complete' if passed else 'parity-mismatch', completed=len(results), total=len(rows), updated=time.time()))
    if not passed:
        raise ValueError('Parity gate failed; inspect retained per-crop evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['export', 'parity', 'all'], default='all')
    args = parser.parse_args()
    args.source_run, args.out = args.source_run.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['HOMR_MODEL_NAME'] = 'public-joint-core-heads-20260928-epoch27-qual20261002'
    setup()
    try:
        if args.phase in ('export', 'all'):
            export(args.source_run, args.out)
        if args.phase in ('parity', 'all'):
            parity(args.source_run, args.out)
    except Exception as error:
        write(args.out / 'ERROR.json', dict(error=str(error), updated=time.time()))
        raise


if __name__ == '__main__':
    main()
