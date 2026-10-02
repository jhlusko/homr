"""Bounded local full-page runtime smoke; this is not ground-truth accuracy qualification."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

from training.onnx.qualify_joint_export import setup, config, sha, write

POLICY = {
    'HOMR_ENFORCE_FINAL_DIVIDER': '0', 'HOMR_TUPLET_REPAIR': '1',
    'HOMR_STEM_ARBITRATION': '0', 'HOMR_SLUR_SIDE': '0',
    'HOMR_BEAM_REPAIR': '1', 'HOMR_TIE_REPAIR': '1', 'HOMR_SLUR_CROSSING': '1',
    'HOMR_CROSS_STAFF_REPAIR': '1', 'HOMR_CROSS_STAFF_POSITION_REPAIR': '1',
    'HOMR_STRUCTURED_EXPORT': 'beam,stem,tie,slur',
}


def process(page, models, arm, out):
    os.environ.update(POLICY)
    setup()
    from homr import main
    from homr.music_xml_generator import XmlGeneratorArguments
    from homr.segmentation.config import segnet_path_onnx
    from training.transformer.qualify_joint_holdout import serialize
    out.mkdir(parents=True, exist_ok=True)
    if sha(page['image']) != page['image_sha256']:
        raise ValueError('Original page changed')
    for record in models[arm].values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Graph changed')
    cfg = config()
    for component, attribute in [('encoder','encoder_path'),('decoder','decoder_path'),('heads','structured_heads_path')]:
        setattr(cfg.filepaths, attribute, models[arm][component]['path'])
    main.Config = lambda: cfg
    original_parse = main.parse_staffs
    def capture(*args, **kwargs):
        result = original_parse(*args, **kwargs)
        write(out/'decoded-staves.json', [[serialize(symbol) for symbol in voice] for voice in result])
        return result
    main.parse_staffs = capture
    image = out / ('page'+Path(page['image']).suffix)
    shutil.copyfile(page['image'], image)
    identity = dict(page=page,arm=arm,models=models[arm],policy=POLICY,
                    segmentation=dict(path=segnet_path_onnx,sha256=sha(segnet_path_onnx)),
                    threads=1,device='CPU',title_detection=False,
                    text_detectors='not run: current boxes are diagnostics only; this is core/heads/segmentation/export smoke',
                    scope='functional local runtime; not paired ground-truth accuracy or final GPU/provider latency')
    write(out/'identity.json',identity)
    tick = time.monotonic()
    try:
        main.process_image(str(image),main.ProcessingConfig(False,False,False,False,-1,False,False,False,False),XmlGeneratorArguments())
        xml = image.with_suffix('.musicxml')
        root = ET.parse(xml).getroot()
        counts = {name:len(root.findall('.//'+name)) for name in ('part','measure','note','rest','beam','stem','tie','slur','dynamics')}
        if not counts['part'] or not counts['note']:
            raise ValueError('Empty MusicXML output')
        if counts['dynamics']:
            raise ValueError('Unexpected automatic dynamics')
        record=dict(status='ok',seconds=time.monotonic()-tick,xml_sha256=sha(xml),counts=counts)
    except Exception:
        record=dict(status='failed',seconds=time.monotonic()-tick,error=traceback.format_exc())
    write(out/'result.json',record)
    if record['status']!='ok':
        raise RuntimeError(record['error'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--page-index',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    pages=json.loads(args.pages.read_text())
    process(pages[args.page_index],json.loads(args.models.read_text()),args.arm,args.out)
