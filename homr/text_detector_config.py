"""Where the Stage 3 text-detector ONNX weights live once downloaded.

Two checkpoints, not one - the "with lyrics" (vocal) and "without lyrics" (instrumental)
toggle decided in `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` ("Shipping decision: two
detectors"). Each is trained with gradient on a different class subset and degrades
outside it, so which one runs is a correctness choice, not a preference - there is no
single "the" detector the way there is a single encoder or decoder.

No inference class reads these paths yet; only `download_weights` does, so both files
are fetched and ready once that wiring lands. Flat module-level constants, matching
`homr/segmentation/config.py`'s own style for the same kind of thing (a model's fixed
on-disk location), rather than folding these into `homr.transformer.configs.Config` -
the text detector is not part of the transformer pipeline.
"""

import os

workspace = os.path.dirname(os.path.realpath(__file__))

#: "With lyrics": trained on Lyrics+Dynamic (Dice), the classes it actually has gradient
#: on. Running an instrumental page through this one produces spurious Lyrics boxes.
detector_vocal_path = os.path.join(workspace, "detector_vocal_e2.onnx")

#: "Without lyrics": trained on Dynamic/Tempo/StaffText/Expression. Running a vocal page
#: through this one loses lyrics entirely - it receives no gradient on that class at all.
detector_instrumental_path = os.path.join(workspace, "detector_instrumental.onnx")
