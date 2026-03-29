"""
ONNX optimized model inference
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

class ONNXModel:
    def __init__(self, model_path="models/resnet50.onnx"):
        """Initialize ONNX Runtime session"""
        self.model_path = str(Path(model_path).resolve())

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                f"Run: python scripts/convert_to_onnx.py"
            )
        
        print(f"Loading ONNX model from {model_path}...")
        start = time.time()
        
        # Configure ONNX Runtime session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4  # Use 4 threads for CPU
        
        # Use CoreML on Apple Silicon (GPU/Neural Engine), fall back to CPU
        available = ort.get_available_providers()
        if 'CoreMLExecutionProvider' in available:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            print("ONNX Runtime: using CoreML execution provider (M-series GPU/Neural Engine)")
        else:
            providers = ['CPUExecutionProvider']

        # Create inference session
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        load_time = time.time() - start
        print(f"ONNX model loaded in {load_time:.2f}s")
    
    def predict(self, input_tensor):
        """
        Run inference on input tensor
        
        Args:
            input_tensor: numpy array of shape [batch_size, 3, 224, 224]
        
        Returns:
            numpy array of shape [batch_size, 1000] (logits)
        """
        # Ensure input is numpy array with correct dtype
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = input_tensor.numpy()
        
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)
        
        # Ensure input has correct shape
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        
        batch_size = input_tensor.shape[0]
        
        try:
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
            
            result = outputs[0]
            
            # Handle different output shapes
            # ONNX might return [batch, 1000] or [batch, 1000, 1, 1]
            if len(result.shape) == 1:
                result = result.reshape(1, -1)
            elif len(result.shape) == 4:  # [batch, 1000, 1, 1]
                result = result.reshape(batch_size, -1)
            elif len(result.shape) == 2:  # [batch, 1000] - correct shape
                pass
            
            return result
            
        except Exception as e:
            print(f"❌ ONNX inference error with batch_size={batch_size}, input_shape={input_tensor.shape}")
            print(f"   Error: {e}")
            raise
    
    def get_model_info(self):
        """Get model metadata"""
        return {
            "model_path": str(self.model_path),
            "model_type": "ONNX",
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers()
        }


class QuantizedONNXModel:
    """
    INT8 dynamically-quantized ONNX model.

    Intentionally uses CPU-only inference. ConvInteger and DynamicQuantizeLinear
    ops inserted by dynamic quantization are not supported by CoreML, which
    causes ORT to create ~50 execution partitions with massive context-switch
    overhead (~93ms vs ~1ms for FP32 CoreML). CPU-only is the correct backend.
    """

    def __init__(self, model_path="models/resnet50_int8.onnx"):
        self.model_path = str(Path(model_path).resolve())

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"INT8 model not found: {model_path}\n"
                f"Run: python scripts/quantize_model.py"
            )

        print(f"Loading INT8 quantized model from {model_path}...")
        start = time.time()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        load_time = time.time() - start
        print(f"INT8 model loaded in {load_time:.2f}s")

    def predict(self, input_tensor):
        """
        Run INT8 inference on input tensor.

        Args:
            input_tensor: numpy array [batch_size, 3, 224, 224]
        Returns:
            numpy array [batch_size, 1000] (logits)
        """
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = input_tensor.numpy()

        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)

        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, 0)

        batch_size = input_tensor.shape[0]

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        result = outputs[0]
        if len(result.shape) == 1:
            result = result.reshape(1, -1)
        elif len(result.shape) == 4:
            result = result.reshape(batch_size, -1)

        return result

    def get_model_info(self):
        return {
            "model_path": str(self.model_path),
            "model_type": "ONNX_INT8",
            "quantization": "dynamic_int8",
            "backend": "cpu",
            "input_name": self.input_name,
            "output_name": self.output_name,
        }