import os
from time import perf_counter

import numpy as np
from PIL import Image

from homr.cross_staff_rerank import rhythm_candidates_for_staff
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import get_decoder
from homr.transformer.encoder_inference import Encoder
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = get_decoder(self.config)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr.
        """
        x = _transform(image=image)

        t0 = perf_counter()

        # Create special tokens
        start_token = np.array([[1]], dtype=np.int64)
        nonote_token = np.array([[0]], dtype=np.int64)

        # Generate context with encoder. The encoder and decoder may run in
        # different precisions (e.g. the CoreML encoder is fp16 while the
        # decoder stays on the fp32 CPU model), so cast the context to the
        # dtype the decoder expects before handing it over.
        context = self.encoder.generate(x)
        context_dtype = np.float16 if self.decoder.fp16 else np.float32
        if context.dtype != context_dtype:
            context = context.astype(context_dtype)

        # Make a prediction using decoder
        out = self.decoder.generate(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )

        eprint(f"Inference Time Tromr: {perf_counter()-t0}")

        return out

    def predict_greedy_with_margins(
        self, image: NDArray
    ) -> tuple[list[EncodedSymbol], list[tuple[int, float]], NDArray]:
        """The cheap half of Phase 1 candidate generation - costs exactly what
        `predict()` costs (one decode pass; `generate_with_rhythm_margins` adds only
        bookkeeping, no extra model calls), unlike forking alternates which each need a
        full extra decode. Returns the *unfiltered* greedy decode (matching the
        decoder's own step numbering `fork_candidates_from_margins`/`rhythm_alternative`
        need to stay aligned with - `predict_best`'s grandstaff/`position != "lower"`
        filter must be applied by the caller, after any forking, not before), its
        per-step rhythm margins, and the encoder context (so a caller that decides
        forking is worth it doesn't need to re-run the encoder).
        """
        x = _transform(image=image)

        start_token = np.array([[1]], dtype=np.int64)
        nonote_token = np.array([[0]], dtype=np.int64)

        t0 = perf_counter()
        context = self.encoder.generate(x)
        context_dtype = np.float16 if self.decoder.fp16 else np.float32
        if context.dtype != context_dtype:
            context = context.astype(context_dtype)

        greedy, margins = self.decoder.generate_with_rhythm_margins(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )
        eprint(f"Inference Time Tromr: {perf_counter()-t0}")

        return greedy, margins, context

    def predict_candidates(self, image: NDArray, max_forks: int = 3) -> list[list[EncodedSymbol]]:
        """
        Phase 1 (decode-time cross-staff-consistency reranking,
        `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2): like `predict()`, but returns the
        greedy decode (`candidates[0]`, identical to what `predict()` alone would
        return) plus up to `max_forks` alternate decodes branched at the rhythm head's
        narrowest-margin decisions (`homr.cross_staff_rerank.rhythm_candidates_for_staff`).
        `parse_staffs` uses this when reranking is enabled, picking among these
        candidates per staff once every staff in a system is available for cross-staff
        comparison; a caller that isn't doing system-level reranking should keep using
        `predict()` directly.
        """
        x = _transform(image=image)

        start_token = np.array([[1]], dtype=np.int64)
        nonote_token = np.array([[0]], dtype=np.int64)

        context = self.encoder.generate(x)
        context_dtype = np.float16 if self.decoder.fp16 else np.float32
        if context.dtype != context_dtype:
            context = context.astype(context_dtype)

        t0 = perf_counter()
        candidates = rhythm_candidates_for_staff(
            self.decoder,
            start_token,
            nonote_token,
            max_forks=max_forks,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )
        eprint(f"Inference Time Tromr (candidates): {perf_counter()-t0}")

        return candidates


class ConvertToArray:
    def __init__(self) -> None:
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray) -> NDArray:
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray) -> NDArray:
        arr = np.array(image) / 255
        arr = arr[np.newaxis, np.newaxis, :, :]
        return self.normalize(arr).astype(np.float32)


_transform = ConvertToArray()


def test_transformer_on_image(path_to_img: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """

    model = Staff2Score(Config())
    image = Image.open(path_to_img)
    out = model.predict(np.array(image))
    eprint(out)


if __name__ == "__main__":
    import sys

    test_transformer_on_image(sys.argv[1])
