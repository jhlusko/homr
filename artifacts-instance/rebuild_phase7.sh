#!/bin/bash
cd /workspace/b0/homr
for split in train valid; do
  ./.venv/bin/python -u -m training.omr_datasets.convert_ossq \
    --dataset-root /workspace/b0/ossq-omr \
    --out /workspace/b0/phase7fix/$split \
    --track scanned --split $split \
    > /workspace/b0/phase7fix_$split.log 2>&1
  echo "SPLIT_${split}_EXIT=$?" >> /workspace/b0/phase7fix_$split.log
done
echo PHASE7FIX_DONE
