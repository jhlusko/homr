import os
import shutil
import sys
from typing import Any

import torch
import torch._dynamo
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR, load_model
from training.omr_datasets.convert_grandstaff import (
    convert_grandstaff,
    grandstaff_train_index,
)
from training.omr_datasets.convert_lieder import convert_lieder, lieder_train_index
from training.omr_datasets.convert_musetrainer import (
    convert_musetrainer,
    musetrainer_train_index,
)
from training.omr_datasets.convert_pdmx import convert_pdmx, pdmx_train_index
from training.omr_datasets.convert_primus import (
    convert_primus_dataset,
    primus_train_index,
)
from training.run_id import get_run_id
from training.transformer.data_loader import label_names, load_dataset, load_dataset_split
from training.transformer.distribute import Distribute
from training.transformer.metrics import HomrTrainer
from training.transformer.mix_datasets import mix_training_sets

torch._dynamo.config.suppress_errors = True


class FreezeCallback(TrainerCallback):
    """
    Callback to freeze the backbone for a set number of epochs.
    Standard practice is ~2 epochs.
    """

    def __init__(self, epochs_to_freeze: int = 2):
        self.epochs_to_freeze = epochs_to_freeze
        self._backbone_frozen = False

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and hasattr(model, "freeze_backbone"):
            eprint(f"Freezing backbone for the first {self.epochs_to_freeze} epochs")
            model.freeze_backbone()
            self._backbone_frozen = True

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        model = kwargs.get("model")
        if model and self._backbone_frozen and state.epoch and state.epoch >= self.epochs_to_freeze:
            eprint(f"Unfreezing backbone at epoch {state.epoch}")
            model.unfreeze_backbone()
            self._backbone_frozen = False


def load_training_index(file_path: str) -> list[str]:
    with open(file_path) as f:
        return f.readlines()


def check_data_source(all_file_paths: list[str]) -> bool:
    result = True
    for file_paths in all_file_paths:
        paths = file_paths.strip().split(",")
        for path in paths:
            if path == "nosymbols":
                continue
            if not os.path.exists(path):
                eprint(f"Index {file_paths} does not exist due to {path}")
                result = False
    return result


def load_and_mix_training_sets(
    index_paths: list[str], weights: list[float], number_of_files: int
) -> list[str]:
    if len(index_paths) != len(weights):
        eprint("Error: Number of index paths and weights do not match")
        sys.exit(1)
    data_sources = [load_training_index(index) for index in index_paths]
    if not all(check_data_source(data) for data in data_sources):
        eprint("Error in datasets found")
        sys.exit(1)
    eprint(
        "Total number of training files to choose from", sum([len(data) for data in data_sources])
    )
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

git_root = os.path.join(script_location, "..", "..")


def _check_datasets_are_present(selected_datasets: list[str]) -> list[str]:
    for dataset in selected_datasets:
        if dataset == primus_train_index and not os.path.exists(primus_train_index):
            convert_primus_dataset()

        if dataset == grandstaff_train_index and not os.path.exists(grandstaff_train_index):
            convert_grandstaff()

        if dataset == lieder_train_index and not os.path.exists(lieder_train_index):
            convert_lieder()

        if dataset == musetrainer_train_index and not os.path.exists(musetrainer_train_index):
            convert_musetrainer()

        if dataset == pdmx_train_index and not os.path.exists(pdmx_train_index):
            convert_pdmx()
    return selected_datasets


def train_transformer(
    fp32: bool = False,
    resume: str = "",
    smoke_test: bool = False,
    fine_tune: bool = False,
    warm_start: bool = False,
    dataset_index: list[str] | None = None,
    dataset_weights: list[float] | None = None,
    number_of_epochs: int | None = None,
    number_of_files: int = -1,
    validation_index: str | None = None,
    checkpoint: str | None = None,
) -> None:
    """`warm_start` starts from the pretrained checkpoint with *every* parameter
    trainable - unlike `fine_tune`, which loads the same checkpoint but freezes
    the encoder and decoder and only unfreezes the lift head. Adapting to a new
    data distribution (real historical scans, as opposed to the synthetic
    renders every existing index is built from) needs the whole model to move,
    not just one head, but starting from scratch would throw away the pretrained
    weights for no reason.

    `dataset_index` overrides the default five-corpus mix. Passing it explicitly
    also *suppresses the auto-download/convert* path: `_check_datasets_are_present`
    will happily spend hours downloading and re-rendering a missing corpus, which
    is right for an interactive full training run and badly wrong for a targeted
    one on indexes the caller already knows exist. A caller that names its own
    indexes gets them verified and nothing else.
    """
    distribute = Distribute()

    if number_of_epochs is None:
        number_of_epochs = 35
        if smoke_test:
            number_of_epochs = 10
        elif fine_tune or warm_start:
            number_of_epochs = 15
    resume_from_checkpoint = None

    checkpoint_folder = "current_training"
    if resume:
        resume_from_checkpoint = os.path.join(git_root, checkpoint_folder, resume)
    elif os.path.exists(os.path.join(git_root, checkpoint_folder)):
        if distribute.is_rank0():
            shutil.rmtree(os.path.join(git_root, checkpoint_folder))

    caller_chose_datasets = dataset_index is not None
    if dataset_index is None:
        dataset_index = [
            lieder_train_index,
            grandstaff_train_index,
            primus_train_index,
            pdmx_train_index,
            musetrainer_train_index,
        ]
    if dataset_weights is None:
        dataset_weights = [1.0] * len(dataset_index)
    if len(dataset_weights) != len(dataset_index):
        eprint("Error: dataset_index and dataset_weights lengths differ")
        sys.exit(1)

    if caller_chose_datasets:
        missing = [path for path in dataset_index if not os.path.exists(path)]
        if missing:
            eprint("Error: named training index does not exist:", missing)
            eprint("Refusing to auto-download - name only indexes that are already built.")
            sys.exit(1)
    elif distribute.is_rank0():
        _check_datasets_are_present(dataset_index)
    distribute.barrier()

    # `number_of_files` is what makes `dataset_weights` mean anything:
    # `mix_training_sets` short-circuits to "concatenate every source" whenever it
    # is negative, ignoring weights entirely, so a mixture with a deliberate ratio
    # (general-data replay against domain-adaptation data, say) has to name a
    # positive total. Set weights proportional to the per-source counts you want
    # and this total to their sum, and each source contributes exactly its target.
    train_index = load_and_mix_training_sets(
        dataset_index,
        dataset_weights,
        number_of_files,
    )

    config = Config()
    if checkpoint is not None:
        # Continue from a specific run's weights rather than the pinned checkpoint.
        # Overridden here rather than on disk: the pinned file is what production and
        # every other run loads, and swapping it to continue one experiment would
        # silently redirect all of them.
        config.filepaths.checkpoint = checkpoint
    if validation_index is not None:
        # A score-disjoint split the caller built (§13.5) - see load_dataset_split
        # for why the default val_split cannot express one.
        if not os.path.exists(validation_index):
            eprint("Error: validation index does not exist:", validation_index)
            sys.exit(1)
        datasets = load_dataset_split(
            train_index, load_training_index(validation_index), config
        )
    else:
        datasets = load_dataset(train_index, config, val_split=0.1)

    compile_threshold = 50000
    compile_model = (
        number_of_files < 0 or number_of_files * number_of_epochs >= compile_threshold
    )  # Compiling needs time, but pays off for large datasets
    if compile_model:
        eprint("Compiling model")

    run_id = get_run_id()

    batch_size = 8  # 8gb vram

    train_args = TrainingArguments(
        checkpoint_folder,
        torch_compile=compile_model,
        eval_strategy="epoch",
        save_strategy="epoch",
        # 1e-4 trains from scratch; 1e-5 is the frozen-backbone fine tune. A warm
        # start sits between them on purpose: every parameter is moving, so the
        # from-scratch rate would wash out the pretrained weights it exists to
        # keep, while the fine-tune rate is slower than a full-model adaptation
        # to a new data distribution needs.
        learning_rate=1e-5 if fine_tune else (3e-5 if warm_start else 1e-4),
        optim="adamw_torch_fused",
        gradient_accumulation_steps=max(1, 4 // distribute.get_world_size()),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        num_train_epochs=number_of_epochs,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join("logs", f"run{run_id}"),
        label_names=label_names,
        bf16=not fp32,
        dataloader_pin_memory=True,
        dataloader_num_workers=12,
    )

    if fine_tune:
        eprint("Fine tuning model from", config.filepaths.checkpoint)
        model = load_model(config)
        model.freeze_encoder()
        model.freeze_decoder()
        model.unfreeze_lift_decoder()
    elif warm_start:
        eprint("Warm starting (all parameters trainable) from", config.filepaths.checkpoint)
        model = load_model(config)
    else:
        model = TrOMR(config)

    model_name = "pytorch_model"

    model_destination = os.path.join(
        git_root, "training", "architecture", "transformer", f"{model_name}_{run_id}.pth"
    )

    if os.path.exists(model_destination):
        eprint("Model already exists", model_destination)
        distribute.destroy()
        return

    try:
        callbacks: list[TrainerCallback] = [EarlyStoppingCallback(early_stopping_patience=5)]
        if not fine_tune:
            callbacks.append(FreezeCallback(epochs_to_freeze=2))

        trainer = HomrTrainer(
            model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            callbacks=callbacks,
            distribute=distribute,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        eprint("Interrupted")
    if distribute.is_rank0():
        torch.save(model.state_dict(), model_destination)
        eprint(f"Saved model to {model_destination}")
    distribute.barrier()
    distribute.destroy()


if __name__ == "__main__":
    if "--fine" in sys.argv:
        train_transformer(fp32=False, fine_tune=True)
    elif len(sys.argv) > 1:
        raise ValueError("Unknown argument")
    else:
        train_transformer(smoke_test=True)
