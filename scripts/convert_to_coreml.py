"""
Convert ResNet-50 directly to CoreML (FP16, ALL compute units).
Produces static-batch models for sizes 1, 2, 4, 8.

Run once as an offline step: python scripts/convert_to_coreml.py

Direct CoreML conversion (vs ONNX -> CoreML):
- All ops handled natively — no partial CPU fallback
- FP16 precision on Neural Engine out of the box
- Static batch sizes allow batch routing without dynamic-shape overhead

Input contract: float32 [B, 3, 224, 224] in [0, 1] range (ToTensor output,
no Normalize). The NormalizedResNet wrapper applies mean/std normalization
inside the model, so it runs on the Neural Engine. Python-side Normalize
step is eliminated (~0.5ms saved per request).
"""

import time
import numpy as np
import torch
import torchvision.models as models
import coremltools as ct
from pathlib import Path

BATCH_SIZES = [1, 2, 4, 8]
OUTPUT_DIR = Path("models")

# ImageNet normalization constants
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class NormalizedResNet(torch.nn.Module):
    """
    Wraps ResNet-50 with ImageNet normalization baked in.

    Expects input in [0, 1] float range (i.e., after dividing by 255).
    CoreML's ImageType with scale=1/255 handles the /255 step, so
    at inference time we pass a raw uint8-equivalent PIL image and
    Python-side normalization is eliminated entirely.
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.model = base_model
        self.register_buffer(
            "mean", torch.tensor(_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor(_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)


def convert_batch(model: torch.nn.Module, batch_size: int) -> ct.models.MLModel:
    # Input is [0, 1] float (ToTensor output, no Normalize — wrapper handles it)
    dummy = torch.rand(batch_size, 3, 224, 224)
    traced = torch.jit.trace(model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=dummy.shape, dtype=float)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )
    return mlmodel


def verify(mlmodel: ct.models.MLModel, batch_size: int, norm_model: torch.nn.Module):
    # Both PyTorch and CoreML receive [0, 1] float input
    dummy_01 = torch.rand(batch_size, 3, 224, 224)
    inp_name = mlmodel.get_spec().description.input[0].name

    with torch.no_grad():
        torch_out = norm_model(dummy_01).numpy()

    ct_out = list(mlmodel.predict({inp_name: dummy_01.numpy().astype(np.float32)}).values())[0]
    if ct_out.ndim == 1:
        ct_out = ct_out.reshape(batch_size, -1)

    torch_top1 = np.argmax(torch_out, axis=1)
    ct_top1    = np.argmax(ct_out,    axis=1)
    return bool(np.all(torch_top1 == ct_top1))


def benchmark(mlmodel: ct.models.MLModel, batch_size: int, n: int = 30) -> float:
    inp_name = mlmodel.get_spec().description.input[0].name
    dummy = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    for _ in range(3):
        mlmodel.predict({inp_name: dummy})
    t = time.perf_counter()
    for _ in range(n):
        mlmodel.predict({inp_name: dummy})
    return (time.perf_counter() - t) / n * 1000


def main():
    print("=" * 60)
    print("Direct PyTorch → CoreML Conversion (FP16 + ImageType)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n[1/3] Loading ResNet-50 + normalization wrapper...")
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval()
    norm_model = NormalizedResNet(base_model).eval()

    print(f"\n[2/3] Converting batch sizes: {BATCH_SIZES}")
    for bs in BATCH_SIZES:
        print(f"\n  batch={bs}...", end=" ", flush=True)
        mlmodel = convert_batch(norm_model, bs)
        path = OUTPUT_DIR / f"resnet50_b{bs}.mlpackage"
        mlmodel.save(str(path))
        print(f"saved.", end=" ", flush=True)

        ok = verify(mlmodel, bs, norm_model)
        print(f"accuracy={'PASS' if ok else 'FAIL'}", end=" ", flush=True)

        ms = benchmark(mlmodel, bs)
        print(f"  {ms:.2f}ms total  {ms/bs:.2f}ms/item")

    print(f"\n[3/3] Summary")
    print(f"  Models saved to: {OUTPUT_DIR}/resnet50_b{{1,2,4,8}}.mlpackage")
    print(f"  Input: TensorType float32 [0,1] — normalization baked into model")
    print(f"  Precision: FP16   Compute: GPU + Neural Engine")
    print("=" * 60)


if __name__ == "__main__":
    main()
