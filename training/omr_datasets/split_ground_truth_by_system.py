"""
Corpus-wide preprocessing: split every OSSQ piece's whole-score ground-truth MusicXML
into small, per-(page, system) fragments.

Why this exists: a per-training-sample lookup against a multi-MB whole-score file -
even cached - does not scale to corpus-wide training. Confirmed empirically:
`phase22`'s first two launch attempts both stalled at 0% GPU utilization with workers
pinned near 100% CPU, and per-worker memory climbing into the multiple-GB range from
caching many large parsed trees (a shuffled dataset means each worker touches many
distinct large pieces before a cache can help). A tiny, pre-extracted,
already-movement-disambiguated fragment is fast to parse *cold*, every time, with no
caching needed at all - this fixes the underlying problem instead of paying to
parallelize around it.

For each piece's real ground truth (`sq<id>.musicxml`) and every (page, system) with
aligned corpus alignment metadata, writes
`metadata/systemwise_ground_truth/<page:04d>:<system:04d>.musicxml` - a window covering
exactly that system's own `measure_start`..`measure_end` range, every part already
carried-forward and movement-disambiguated (`extract_ground_truth_window`, the same
machinery `build_review_assets.py`/`time_signature_for_sample` already use per-sample,
run once here instead).
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from training.omr_datasets.ossq_ground_truth import (
    _systemwise_entries_cached,
    extract_ground_truth_window,
    fragment_path,
)


def _is_real_score_id(stem: str) -> bool:
    """`scores/*/*/sq*.musicxml` also matches non-canonical variants some pieces carry
    alongside the real file (e.g. a `sq<id>_cleaned.musicxml`) - these never match any
    real corpus metadata (which is keyed to the exact `sq<id>` score_id) so they'd only
    waste an iteration, not produce wrong output, but filtering them out here confirms
    the piece count this script reports is the real one, not inflated by variants."""
    return stem.startswith("sq") and stem[2:].isdigit()


def _split_piece_task(args: tuple[str, str, str]) -> tuple[str, int, int]:
    """`ProcessPoolExecutor`-friendly wrapper - takes/returns only picklable plain
    values (str paths, not `Path` objects bound to this process's imports)."""
    gt_path_str, piece_dir_str, score_id = args
    written, skipped = split_piece(Path(gt_path_str), Path(piece_dir_str), score_id)
    return score_id, written, skipped


def split_piece(gt_path: Path, piece_dir: Path, score_id: str) -> tuple[int, int]:
    """Writes one fragment per (page, system) this piece's aligned metadata covers.
    Returns (written, skipped)."""
    entries = _systemwise_entries_cached(str(piece_dir), score_id)
    if not entries:
        return 0, 0

    written = 0
    skipped = 0
    movement = 0
    prev_end: int | None = None
    for page, system_num, start, end in entries:
        # Mirrors movement_index_for_system's own reset-counting exactly - entries are
        # already sorted in page order, the same sequence that function walks.
        if prev_end is not None and start < prev_end:
            movement += 1
        prev_end = end

        out_path = fragment_path(piece_dir, page, system_num)
        try:
            ok = extract_ground_truth_window(gt_path, movement, start, end, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR page {page} system {system_num}: {e}", flush=True)
            skipped += 1
            continue
        if ok:
            written += 1
        else:
            skipped += 1
    return written, skipped


def main() -> None:
    dataset_root = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    all_matches = sorted(dataset_root.glob("scores/*/*/sq*.musicxml"))
    scores = [gt for gt in all_matches if _is_real_score_id(gt.stem)]
    skipped_variants = len(all_matches) - len(scores)
    print(
        f"{len(scores)} pieces found under {dataset_root} "
        f"({skipped_variants} non-canonical-filename variants ignored), workers={workers}",
        flush=True,
    )

    total_written = 0
    total_skipped = 0
    t0 = time.time()

    if workers <= 1:
        for idx, gt_path in enumerate(scores, 1):
            score_id = gt_path.stem
            piece_dir = gt_path.parent
            written, skipped = split_piece(gt_path, piece_dir, score_id)
            total_written += written
            total_skipped += skipped
            if idx % 20 == 0 or idx == len(scores):
                elapsed = time.time() - t0
                print(
                    f"[{idx}/{len(scores)}] {score_id}: {written} written, {skipped} skipped "
                    f"(total {total_written} written, {total_skipped} skipped, {elapsed:.1f}s elapsed)",
                    flush=True,
                )
    else:
        # Pieces are fully independent - embarrassingly parallel across processes.
        # This corpus has 122 real pieces / ~10,400 systems; serial (~0.3-0.5s per
        # system, each `extract_ground_truth_window` call deliberately re-parses its
        # piece's whole-score file fresh rather than risking a shared mutable cache -
        # see its own docstring) was measured at ~70 minutes total. With this
        # instance's 128 idle vCPUs (confirmed via `nproc`/`nvidia-smi` while
        # diagnosing `phase22`'s CPU-bound stall), parallelizing across pieces cuts
        # that to a few minutes.
        tasks = [(str(gt), str(gt.parent), gt.stem) for gt in scores]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_split_piece_task, task): task for task in tasks}
            for future in as_completed(futures):
                score_id, written, skipped = future.result()
                total_written += written
                total_skipped += skipped
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    print(
                        f"[{done}/{len(tasks)}] {score_id}: {written} written, {skipped} skipped "
                        f"(total {total_written} written, {total_skipped} skipped, {elapsed:.1f}s elapsed)",
                        flush=True,
                    )

    print(f"\n===== SPLIT SUMMARY =====")
    print(f"pieces: {len(scores)}")
    print(f"fragments written: {total_written}")
    print(f"fragments skipped (no measure in range / error): {total_skipped}")


if __name__ == "__main__":
    main()
