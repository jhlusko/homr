"""The scheduled-sampling first pass must not starve the second pass of gradients.

Regression test for E1 run 1 (2026-09-25): under CUDA bf16 autocast, the no-grad first
pass cached weight casts without autograd links, so the real pass trained only the
embeddings. CUDA-only: CPU autocast does not cache weight casts this way.
"""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "the autocast weight cache is CUDA-only")
class TestScheduledSamplingGradients(unittest.TestCase):
    def test_output_heads_and_attention_get_gradients_after_a_no_grad_first_pass(self) -> None:
        from homr.transformer.configs import Config
        from training.architecture.transformer.tromr_arch import TrOMR

        torch.manual_seed(0)
        config = Config()
        model = TrOMR(config).cuda().train()
        decoder = model.decoder
        batch, length = 2, 12
        tokens = {
            name: torch.ones(batch, length, dtype=torch.long, device="cuda")
            for name in ("rhythms", "pitchs", "lifts", "articulations", "slurs", "positions")
        }
        mask = torch.ones(batch, length, dtype=torch.bool, device="cuda")
        image = torch.randn(batch, config.channels, config.max_height, config.max_width, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = decoder(**tokens, mask=mask, sampling_prob=0.5, context=model.encoder(image))
        out["loss"].backward()
        self.assertIsNotNone(decoder.net.to_logits_rhythm.weight.grad)
        self.assertGreater(decoder.net.to_logits_rhythm.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
