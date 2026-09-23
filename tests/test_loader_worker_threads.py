"""A forked loader worker that keeps torch's default thread pool deadlocks the loader."""

import unittest

import torch

from training.architecture.transformer.train_structured_heads import (
    _cap_worker_threads,
    build_batches,
)


class TestCapWorkerThreads(unittest.TestCase):
    def test_it_holds_the_worker_to_one_intra_op_thread(self) -> None:
        before = torch.get_num_threads()
        try:
            _cap_worker_threads(0)
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(before)

    def test_the_loader_installs_it_only_when_workers_are_forked(self) -> None:
        # With num_workers=0 there is no fork and no worker to initialise, and passing a
        # worker_init_fn there would be a claim the loader does not honour.
        import inspect

        source = inspect.getsource(build_batches)
        self.assertIn("worker_init_fn=_cap_worker_threads if workers else None", source)


if __name__ == "__main__":
    unittest.main()
