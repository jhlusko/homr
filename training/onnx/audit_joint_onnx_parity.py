"""Audit retained parity differences and isolate encoder versus attention arithmetic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from training.onnx.qualify_joint_export import config, setup, sha, write


def audit(source, out):
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments
    from training.transformer.review_cached_joint_decoding import restore
    from training.transformer.qualify_joint_holdout import serialize
    from training.transformer.training_vocabulary import read_tokens
    from training.transformer.audit_cached_joint_decoding import projection
    from rapidfuzz.distance import Levenshtein
    rows = [json.loads(line) for line in (out / 'parity.jsonl').open()]
    totals = {'eager': Counter(), 'onnx': Counter()}
    cases = []
    directory = out / 'xml-parity'
    directory.mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        reference = [serialize(s) for s in read_tokens(row['tokens'])]
        differences = []
        for step, (a, b) in enumerate(zip(row['eager'], row['onnx'])):
            if a != b:
                differences.append(dict(step=step, rhythm=a['rhythm'],
                    fields={k: [a.get(k), b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}))
        xmls = {}
        errors = {}
        for arm in ('eager', 'onnx'):
            for metric in ('symbol', 'rhythm', 'pitch', 'accidental'):
                a, b = projection(reference, metric), projection(row[arm], metric)
                totals[arm][metric+'_errors'] += Levenshtein.distance(a,b)
                totals[arm][metric+'_support'] += len(a)
            try:
                xmls[arm] = ET.tostring(generate_xml(XmlGeneratorArguments(), [restore(row[arm])], ''), encoding='unicode')
                (directory / f'{index:02}-{arm}.musicxml').write_text(xmls[arm])
            except Exception as error:
                errors[arm] = str(error)
        cases.append(dict(id=row['id'], score=row['score'], recognition_equal=row['recognition_equal'],
            xml_equal=not errors and xmls['eager']==xmls['onnx'], xml_errors=errors, differences=differences))
    write(out / 'semantic-parity-audit.json', dict(crops=len(rows),
        recognition_equal=sum(c['recognition_equal'] for c in cases),
        xml_equal=sum(c['xml_equal'] for c in cases), totals={k:dict(v) for k,v in totals.items()}, cases=cases,
        limitation='Original strict parity failure is retained. XML equality diagnoses whether raw field differences are observable; it does not silently redefine the gate. No v7 confidence policy applied.'))
    return rows


def isolate(source, out, row):
    import torch
    from training.architecture.transformer.tromr_arch import TrOMR
    from homr.transformer.staff2score import Staff2Score
    from training.transformer.qualify_joint_holdout import pixels, serialize
    import x_transformers.attend
    artifacts = json.loads((out/'exports.json').read_text())
    spec = json.loads((source/'models.json').read_text())['public_joint27']
    if sha(spec['joint']) != spec['sha256']['joint']:
        raise ValueError('Checkpoint changed')
    cfg = config()
    for component, attribute in [('encoder','encoder_path'),('decoder','decoder_path'),('heads','structured_heads_path')]:
        setattr(cfg.filepaths, attribute, artifacts[component]['path'])
    model = TrOMR(cfg).eval()
    model.load_state_dict(torch.load(spec['joint'],map_location='cpu',weights_only=True),strict=True)
    model.decoder.device = torch.device('cpu')
    onnx = Staff2Score(cfg)
    tensor = pixels(row['image']).unsqueeze(0)
    torch_logits = []
    def hook(module, args, output):
        torch_logits.append(output[0][0,-1,:].detach().numpy().copy())
    handle = model.decoder.net.register_forward_hook(hook)
    with torch.inference_mode():
        context = model.encoder(tensor).numpy()
        eager = [serialize(s) for s in model.generate(tensor)]
    handle.remove()
    ort_logits = []
    run = onnx.decoder.net.run_with_iobinding
    def capture(*args, **kwargs):
        result = run(*args, **kwargs)
        ort_logits.append(onnx.decoder.io_binding.get_outputs()[0].numpy()[0,-1,:].copy())
        return result
    onnx.decoder.net.run_with_iobinding = capture
    def decode(ctx):
        return [serialize(s) for s in onnx.decoder.generate(np.array([[1]],dtype=np.int64),
            np.array([[0]],dtype=np.int64),seq_len=cfg.max_seq_len,eos_token=cfg.eos_token,context=ctx)]
    native_onnx = decode(onnx.encoder.generate(tensor.numpy()))
    baseline_logits = ort_logits[:]
    same_context_onnx = decode(context)
    altered = 0
    for module in model.decoder.net.modules():
        if isinstance(module,x_transformers.attend.Attend) and module.flash:
            module.flash=False
            altered += 1
    with torch.inference_mode():
        manual_eager = [serialize(s) for s in model.generate(tensor)]
    tokens = lambda rows: [{k:r[k] for k in ('rhythm','pitch','lift','articulation','slur','position')} for r in rows]
    steps=[]
    for step,(a,b) in enumerate(zip(eager,native_onnx)):
        if a['rhythm']!=b['rhythm']:
            def top(values):
                ids=np.argsort(values)[-3:][::-1]
                return dict(values=[dict(token=cfg.rhythm_vocab_inv[int(i)] if hasattr(cfg,'rhythm_vocab_inv') else next(k for k,v in cfg.rhythm_vocab.items() if v==int(i)),logit=float(values[i])) for i in ids],margin=float(values[ids[0]]-values[ids[1]]))
            steps.append(dict(step=step,eager=top(torch_logits[step]),onnx=top(baseline_logits[step])))
    write(out/'attention-isolation.json',dict(id=row['id'],score=row['score'],
        runtime_encoder_equals_eager_tokens=tokens(same_context_onnx)==tokens(eager),
        manual_eager_equals_onnx_tokens=tokens(manual_eager)==tokens(native_onnx),
        altered_attention_modules=altered, rhythm_differences=steps,
        eager=eager,onnx=native_onnx,onnx_same_encoder_context=same_context_onnx,manual_eager=manual_eager))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    setup()
    rows=audit(args.source_run,args.out)
    failures=[r for r in rows if not r['recognition_equal']]
    if failures:
        isolate(args.source_run,args.out,failures[0])
