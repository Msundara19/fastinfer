"""
Direct CoreML inference models — bypasses ONNX entirely.

Single image: uses the batch=1 FP16 model (1.17ms vs 1.33ms ONNX FP32).

Batched: StaticBatchCoreMLModel routes each batch to the closest
pre-compiled static model (b1/b2/b4/b8), padding as needed.
This fixes the CoreML dynamic-shape fallback that caused the ONNX
batched endpoint to drop from 85 to 12 req/s.

Per-item latency at each batch size:
  b1: 1.17ms  b2: 0.95ms  b4: 0.84ms (sweet spot)  b8: 1.01ms

Input contract (after running convert_to_coreml.py):
  Models use NormalizedResNet wrapper — mean/std normalization runs on
  the Neural Engine. Pass float32 [0,1] arrays (ToTensor output, no
  Normalize). Python-side Normalize step is eliminated.
"""

import time
import numpy as np
import coremltools as ct
from pathlib import Path


# Static batch sizes available — must match compiled .mlpackage files
_BATCH_SIZES = [1, 2, 4, 8]


class CoreMLModel:
    """Single-image CoreML inference (FP16, Neural Engine)."""

    def __init__(self, model_path: str = "models/resnet50_b1.mlpackage"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"CoreML model not found: {model_path}\n"
                f"Run: python scripts/convert_to_coreml.py"
            )
        print(f"Loading CoreML model from {model_path}...")
        start = time.time()
        self._model = ct.models.MLModel(str(path))
        self._input_name = self._model.get_spec().description.input[0].name
        print(f"CoreML model loaded in {time.time() - start:.2f}s")

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        input_array: float32 [1, 3, 224, 224] in [0, 1]  →  float32 [1, 1000]

        Models use NormalizedResNet wrapper — mean/std normalization runs on
        the Neural Engine. Pass ToTensor output (no Normalize in Python).
        """
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)
        out = self._model.predict({self._input_name: input_array})
        result = list(out.values())[0]
        if result.ndim == 1:
            result = result.reshape(1, -1)
        return result

    def get_model_info(self) -> dict:
        return {
            "model_type": "CoreML_FP16",
            "compute_units": "ALL (GPU + Neural Engine)",
            "precision": "float16",
        }


class StaticBatchCoreMLModel:
    """
    Batched CoreML inference using pre-compiled static-shape models.

    Routes a batch of N images to the smallest compiled model that fits
    (b1/b2/b4/b8), padding with zeros if N doesn't match exactly, then
    discards the padded outputs.
    """

    def __init__(self, model_dir: str = "models"):
        self._models: dict[int, ct.models.MLModel] = {}
        self._input_names: dict[int, str] = {}

        print("Loading static-batch CoreML models...")
        start = time.time()
        for bs in _BATCH_SIZES:
            path = Path(model_dir) / f"resnet50_b{bs}.mlpackage"
            if not path.exists():
                raise FileNotFoundError(
                    f"CoreML batch model not found: {path}\n"
                    f"Run: python scripts/convert_to_coreml.py"
                )
            m = ct.models.MLModel(str(path))
            self._models[bs] = m
            self._input_names[bs] = m.get_spec().description.input[0].name

        print(f"Static-batch CoreML models loaded in {time.time() - start:.2f}s "
              f"(batch sizes: {_BATCH_SIZES})")

    def _target_batch(self, n: int) -> int:
        """Return the smallest compiled batch size that fits n."""
        for bs in _BATCH_SIZES:
            if bs >= n:
                return bs
        return _BATCH_SIZES[-1]  # chunk into max-size batches if n > 8

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        input_array: float32 [N, 3, 224, 224] in [0, 1]
        Returns:     float32 [N, 1000]
        """
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)

        n = input_array.shape[0]

        if n > _BATCH_SIZES[-1]:
            # Chunk large batches
            results = []
            for i in range(0, n, _BATCH_SIZES[-1]):
                chunk = input_array[i:i + _BATCH_SIZES[-1]]
                results.append(self.predict(chunk))
            return np.concatenate(results, axis=0)

        target = self._target_batch(n)
        model = self._models[target]
        input_name = self._input_names[target]

        # Pad if needed (zeros = black pixels, discarded after inference)
        if n < target:
            pad = np.zeros((target - n, *input_array.shape[1:]), dtype=input_array.dtype)
            padded = np.concatenate([input_array, pad], axis=0)
        else:
            padded = input_array

        out = model.predict({input_name: padded})
        result = list(out.values())[0]
        if result.ndim == 1:
            result = result.reshape(target, -1)

        return result[:n]  # discard padded outputs

    def get_model_info(self) -> dict:
        return {
            "model_type": "CoreML_StaticBatch_FP16",
            "batch_sizes": _BATCH_SIZES,
            "compute_units": "ALL (GPU + Neural Engine)",
            "precision": "float16",
        }
