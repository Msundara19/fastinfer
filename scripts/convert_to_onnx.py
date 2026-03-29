"""
Convert PyTorch ResNet-50 to ONNX format
"""

import torch
import torchvision.models as models
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np

def convert_resnet50_to_onnx(output_path="models/resnet50.onnx"):
    """
    Convert pretrained ResNet-50 to ONNX format
    """
    print("🔄 Starting PyTorch → ONNX conversion...")
    
    # Create models directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model
    print("📥 Loading pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    print(f"💾 Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # Use opset 17 for better compatibility
        do_constant_folding=True,  # Optimize constant folding
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},   # Variable batch size
            'output': {0: 'batch_size'}
        }
    )
    
    # Merge external data into a single self-contained file
    # (required for CoreML execution provider)
    data_file = Path(output_path + ".data")
    if data_file.exists():
        print("🔗 Merging external weights into single-file model...")
        onnx_model = onnx.load(output_path, load_external_data=True)
        onnx.save(onnx_model, output_path, save_as_external_data=False)
        data_file.unlink()
        print("✅ Weights merged into single ONNX file")

    # Verify ONNX model
    print("✅ Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test ONNX Runtime inference
    print("🧪 Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # Run test inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare PyTorch vs ONNX outputs
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    
    np.testing.assert_allclose(
        pytorch_output.numpy(), 
        ort_outputs[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    
    print("✅ ONNX model verified! Outputs match PyTorch.")
    
    # Print model info
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n📊 Model Information:")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Input shape: [batch_size, 3, 224, 224]")
    print(f"  Output shape: [batch_size, 1000]")
    print(f"  Opset version: 17")
    print(f"  Dynamic batching: Enabled")
    
    print(f"\n✅ Conversion complete! ONNX model saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    convert_resnet50_to_onnx()