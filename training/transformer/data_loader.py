import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import Vocabulary
from training.architecture.transformer.profile_context import (
    ProfileContext,
    apply_context_dropout,
    context_to_batch_fields,
)
from training.omr_datasets.score_profile_pairing import profile_and_part_for_sample
from training.omr_datasets.score_profile_time_signature import time_signature_for_sample
from training.transformer.image_utils import (
    distort_image,
    ndarray_to_tensor,
    pad_to_3_dims,
    prepare_for_tensor,
    read_image_to_ndarray,
)
from training.transformer.training_vocabulary import (
    DecoderBranches,
    max_ledger_lines,
    read_tokens,
    to_decoder_branches,
)

script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")

label_names = ["rhythms", "positions", "lifts", "pitchs", "articulations", "slurs"]

MAX_ALLOWED_LEDGER_LINES = 5


class DataLoader:
    def __init__(
        self,
        corpus_list: list[str],
        config: Config,
        is_validation: bool = False,
        dataset_root: str | None = None,
    ) -> None:
        self.corpus_list = self._add_mask_steps(corpus_list)
        self.vocab = Vocabulary()
        self.config = config
        self.is_validation = is_validation
        # §7.3/§7.4 score-profile conditioning - None (the default) preserves the
        # existing behaviour exactly: no profile_* keys are added to a sample at all,
        # so a caller that never asks for this sees no difference. Only OSSQ samples
        # currently resolve to anything (training.omr_datasets.score_profile_pairing);
        # every other corpus mixed into the same corpus_list (mix_datasets.py routinely
        # mixes several) falls through to the same "no profile" case a genuinely
        # unresolvable OSSQ sample does.
        self.dataset_root = dataset_root

    def _add_mask_steps(self, corpus_list: list[str]) -> Any:
        result = []
        for entry in corpus_list:
            image, token_file = entry.strip().split(",")
            result.append({"image": image, "tokens": token_file})
        return result

    def __len__(self) -> int:
        return len(self.corpus_list)

    def _read_tokens(self, path: str) -> DecoderBranches:
        tokens = read_tokens(path)
        return to_decoder_branches(tokens)

    def __getitem__(self, idx: int) -> Any:
        entry = self.corpus_list[idx]
        sample_filepath = os.path.join(git_root, entry["image"])

        try:
            img = read_image_to_ndarray(sample_filepath)
        except ValueError:
            eprint(f"Warning: skipping corrupted image {sample_filepath}")
            return self.__getitem__((idx + 1) % len(self))

        if self.is_validation:
            state = random.getstate()
            np_state = np.random.get_state()
            random.seed(idx)
            np.random.seed(idx)

        img = distort_image(img, allow_occlusions=not self.is_validation)
        img = add_image_into_tr_omr_canvas(img)

        if self.dataset_root is not None:
            # Resolved (and, for training, dropped-out per §7.4) inside the same
            # seeded/restored block as the image distortion above - validation gets a
            # reproducible profile_context per idx across runs, same as the image
            # augmentation already does, and training's dropout roll uses this call's
            # live global `random` state rather than a separate generator, so it varies
            # normally epoch to epoch the same way image distortion already does.
            context = self._resolve_profile_context(entry["tokens"])
            if not self.is_validation:
                context = apply_context_dropout(context, random)

        if self.is_validation:
            random.setstate(state)
            np.random.set_state(np_state)

        img = prepare_for_tensor(img)
        sample_img = ndarray_to_tensor(img)
        sample_img = pad_to_3_dims(sample_img)

        tokens_full_filepath = entry["tokens"]
        tokens = self._read_tokens(tokens_full_filepath)

        # Remember to extend label_names if you add something to the results
        result = {
            "inputs": sample_img,
            "rhythms": tokens.rhythms,
            "pitchs": tokens.pitchs,
            "lifts": tokens.lifts,
            "positions": tokens.positions,
            "articulations": tokens.articulations,
            "slurs": tokens.slurs,
            "mask": tokens.mask,
        }
        if self.dataset_root is not None:
            result.update(context_to_batch_fields(context))
        return result

    def _resolve_profile_context(self, tokens_path: str) -> "ProfileContext | None":
        assert self.dataset_root is not None  # only called when it is set
        stem = Path(tokens_path).stem
        resolved = profile_and_part_for_sample(self.dataset_root, stem)
        if resolved is None:
            return None
        profile, part = resolved
        context = ProfileContext.from_score_part(part, part_ordinal=profile.parts.index(part))
        # DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's refinement: the real ground-truth
        # time signature in effect at this sample's own measure range, not just the
        # part-level fields from_score_part already carries - "" (unknown) when it
        # can't be resolved (a non-aligned page/system, an ambiguous multi-movement
        # case) leaves the context exactly as from_score_part built it.
        time_signature = time_signature_for_sample(self.dataset_root, stem)
        if time_signature:
            context = replace(context, expected_time_signature=time_signature)
        return context


def _filter_valid_samples(samples: list[str]) -> list[str]:
    # Filter out samples whose notes require more than the allowed number of ledger lines.
    # MuseScore in these cases might decide to change the clef and then we get a mismatch
    # between what is visible on the image and the ground truth
    valid_samples: list[str] = []
    filter_count = 0
    too_long_count = 0
    for entry in samples:
        image, token_file = entry.strip().split(",")
        tokens = read_tokens(token_file)
        if len(tokens) > default_config.max_seq_len - 2:
            too_long_count += 1
            continue
        if max_ledger_lines(tokens) > MAX_ALLOWED_LEDGER_LINES:
            filter_count += 1
            continue
        valid_samples.append(entry)

    if filter_count > 0:
        eprint(f"Filtered out {filter_count} samples with too many ledger lines")
    if too_long_count > 0:
        eprint(f"Filtered out {too_long_count} samples with sequences longer than max_seq_len")
    return valid_samples


def load_dataset(
    samples: list[str],
    config: Config,
    val_split: float = 0.0,
    dataset_root: str | None = None,
) -> dict[str, Any]:
    samples = _filter_valid_samples(samples)
    val_idx = int(len(samples) * val_split)
    training_list = samples[val_idx:]
    validation_list = samples[:val_idx]

    eprint(
        "Training with "
        + str(len(training_list))
        + " and validating with "
        + str(len(validation_list))
    )
    return {
        "train": DataLoader(
            training_list,
            config,
            is_validation=False,
            dataset_root=dataset_root,
        ),
        "train_list": training_list,
        "validation": DataLoader(
            validation_list,
            config,
            is_validation=True,
            dataset_root=dataset_root,
        ),
        "validation_list": validation_list,
    }
