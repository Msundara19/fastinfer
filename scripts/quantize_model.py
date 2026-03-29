"""
Convert FP32 ONNX model to INT8 dynamic quantization.
Run once as an offline step before starting the server.

Usage: python scripts/quantize_model.py
"""

import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_resnet50(
    input_path="models/resnet50.onnx",
    output_path="models/resnet50_int8.onnx"
):
    print("=" * 60)
    print("INT8 Dynamic Quantization")
    print("=" * 60)

    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"FP32 model not found: {input_path}\n"
            f"Run: python scripts/convert_to_onnx.py"
        )

    # Strip value_info before quantizing.
    # The dynamo exporter stores intermediate shape annotations that conflict
    # with what ONNX shape inference recomputes for the dynamic batch axis.
    # quantize_dynamic always reruns shape inference internally, which raises
    # InferenceError if those annotations are present. Stripping them is safe.
    print(f"\n[1/4] Preparing model...")
    model = onnx.load(input_path)
    del model.graph.value_info[:]
    stripped_path = "/tmp/resnet50_stripped.onnx"
    onnx.save(model, stripped_path)
    print(f"      Stripped {len(model.graph.value_info)} cached shape annotations")

    # Dynamic INT8 quantization — weights quantized at conversion time,
    # activations quantized at runtime. No calibration data required.
    # Only Conv and Gemm are quantized; other ops stay FP32.
    # Do NOT use CoreML provider with this model — ConvInteger and
    # DynamicQuantizeLinear ops are unsupported by CoreML, causing 50
    # execution partitions and ~93ms overhead vs ~1ms for FP32 CoreML.
    print(f"\n[2/4] Quantizing weights to INT8...")
    quantize_dynamic(
        model_input=stripped_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "Gemm"],
        per_channel=False,
        reduce_range=False,  # Not needed on ARM/Apple Silicon
    )

    fp32_mb = Path(input_path).stat().st_size / (1024 * 1024)
    int8_mb = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - int8_mb / fp32_mb) * 100
    print(f"      FP32 size: {fp32_mb:.1f} MB")
    print(f"      INT8 size: {int8_mb:.1f} MB  ({reduction:.0f}% reduction)")

    # Verify correctness against the test image
    print(f"\n[3/4] Verifying accuracy on test_dog.jpg...")
    _verify_accuracy(input_path, output_path)

    print(f"\n[4/4] Done.")
    print(f"      Model saved to: {output_path}")
    print(f"\n      Note: this model uses CPU-only inference.")
    print(f"      CoreML is intentionally skipped (unsupported INT8 ops).")
    print("=" * 60)


def _verify_accuracy(fp32_path, int8_path):
    from PIL import Image
    import torchvision.transforms as transforms
    import torch

    test_image = "test_dog.jpg"
    if not Path(test_image).exists():
        print(f"      Skipping — {test_image} not found")
        return

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(test_image).convert("RGB")
    tensor = transform(img).unsqueeze(0).numpy()

    # FP32 inference
    fp32_session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    fp32_out = fp32_session.run(None, {fp32_session.get_inputs()[0].name: tensor})[0]
    fp32_probs = np.exp(fp32_out[0] - fp32_out[0].max())
    fp32_probs /= fp32_probs.sum()
    fp32_class = int(np.argmax(fp32_probs))
    fp32_conf = float(fp32_probs[fp32_class])

    # INT8 inference
    int8_session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    int8_out = int8_session.run(None, {int8_session.get_inputs()[0].name: tensor})[0]
    int8_probs = np.exp(int8_out[0] - int8_out[0].max())
    int8_probs /= int8_probs.sum()
    int8_class = int(np.argmax(int8_probs))
    int8_conf = float(int8_probs[int8_class])

    top1_match = fp32_class == int8_class
    status = "PASS" if top1_match else "FAIL"
    print(f"      FP32 top-1: class {fp32_class} (conf {fp32_conf:.4f})")
    print(f"      INT8 top-1: class {int8_class} (conf {int8_conf:.4f})")
    print(f"      Top-1 match: {status}")

    if not top1_match:
        raise RuntimeError(
            f"INT8 accuracy check failed: FP32={fp32_class}, INT8={int8_class}"
        )


if __name__ == "__main__":
    quantize_resnet50()
