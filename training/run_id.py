import os


def get_run_id() -> str:
    """Identify a run by the commit it was trained at.

    HOMR_RUN_SUFFIX is appended when set. Two runs started from the same checkout share a
    HEAD and therefore a run id, so they write the same `pytorch_model_<run_id>.pth` - and
    `train_transformer` returns early when that file exists, which once made a "retrain"
    finish in 14 seconds. That is harmless while runs are sequential and fatal once they
    are concurrent, so a replicate distinguishes its seeds with a suffix rather than by
    making an empty commit per seed and racing to read HEAD before the next one lands.
    """
    git_count = os.popen("git rev-list --count HEAD").read().strip()  # noqa: S605, S607
    git_head = os.popen("git rev-parse HEAD").read().strip()  # noqa: S605, S607
    run_id = f"{git_count}-{git_head}"
    suffix = os.environ.get("HOMR_RUN_SUFFIX", "").strip()
    return f"{run_id}-{suffix}" if suffix else run_id
