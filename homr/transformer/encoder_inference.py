import numpy as np
import onnxruntime as ort

from homr.onnx_providers import (
    coreml_available,
    coreml_mlprogram_providers,
    gpu_providers,
)
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.type_definitions import NDArray


class Encoder:
    def __init__(self, config: Config) -> None:
        self.use_gpu = False
        if config.use_gpu_inference:
            try:
                providers, device = gpu_providers({"cudnn_conv_algo_search": "DEFAULT"})
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    providers=providers,
                )
                self.fp16 = True
                # CoreML binds IO on the CPU even though compute runs on the GPU/ANE.
                self.use_gpu = device == "cuda"

            except Exception as ex:
                eprint(ex)
                eprint("Going on without GPU support")
                self.encoder = ort.InferenceSession(config.filepaths.encoder_path_fp16)
                self.fp16 = True

        elif config.use_coreml_encoder and coreml_available():
            try:
                # CPUAndGPU skips the (slow) ANE specialization: it halves the
                # session creation time vs "ALL" and inference is even slightly
                # faster (measured on an M1).
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    providers=coreml_mlprogram_providers(
                        config.filepaths.encoder_path_fp16, compute_units="CPUAndGPU"
                    ),
                )
                self.fp16 = True
                # use_gpu stays False: CoreML binds IO on the CPU even though
                # the compute runs on the GPU/ANE.
            except Exception as ex:
                eprint(ex)
                eprint("Could not create the CoreML encoder session, using the CPU instead")
                self.encoder = ort.InferenceSession(config.filepaths.encoder_path)
                self.fp16 = False

        else:
            self.encoder = ort.InferenceSession(config.filepaths.encoder_path)
            self.fp16 = False

        # The graph is authoritative about its own precision; see the same fix in
        # `decoder_inference.get_decoder`. A fine-tuned export may keep the historical
        # `_fp16` filename while exposing float32 inputs, and binding float16 from the
        # name alone fails before the first op runs. Set after every branch above, so
        # it corrects whichever one ran.
        inputs = self.encoder.get_inputs()
        model_input_type = inputs[0].type if inputs else None
        if model_input_type is None:
            eprint("Encoder graph describes no inputs; keeping the filename's precision.")
        elif model_input_type not in {"tensor(float)", "tensor(float16)"}:
            raise RuntimeError(f"Unsupported HOMR encoder input type: {model_input_type}")
        else:
            self.fp16 = model_input_type == "tensor(float16)"

        self.io_binding = self.encoder.io_binding()
        self.device_id = 0

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> NDArray:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", x.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", x.astype(np.float32))

        self.io_binding.bind_output("output", "cuda" if self.use_gpu else "cpu", self.device_id)
        self.encoder.run_with_iobinding(self.io_binding)
        return self.io_binding.get_outputs()[0].numpy()
