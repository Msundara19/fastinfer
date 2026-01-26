"""
Dynamic batching for improved throughput
Collects requests and processes them in batches
"""

import asyncio
import time
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass
from collections import deque
import torch

@dataclass
class BatchRequest:
    """Individual request in batch"""
    request_id: str
    input_tensor: np.ndarray
    future: asyncio.Future
    timestamp: float

class DynamicBatcher:
    """
    Dynamic batching engine that collects requests and processes them in batches
    """
    
    def __init__(
        self,
        model,
        max_batch_size: int = 8,
        max_wait_ms: int = 10,
        use_onnx: bool = False
    ):
        """
        Initialize dynamic batcher
        
        Args:
            model: Model instance (PyTorch or ONNX)
            max_batch_size: Maximum batch size
            max_wait_ms: Maximum wait time before processing batch (milliseconds)
            use_onnx: Whether using ONNX model
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds
        self.use_onnx = use_onnx
        
        self.queue = deque()
        self.processing = False
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0,
            'total_wait_time': 0
        }
        
        # Start background batch processor
        self.processor_task = None
        
    async def start(self):
        """Start the batch processor"""
        if self.processor_task is None:
            self.processor_task = asyncio.create_task(self._batch_processor())
            print(f"✓ Dynamic batcher started (max_batch={self.max_batch_size}, max_wait={self.max_wait_ms*1000}ms)")
    
    async def stop(self):
        """Stop the batch processor"""
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
            print("✓ Dynamic batcher stopped")
    
    async def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Add request to batch queue and wait for result
        
        Args:
            input_tensor: Input tensor [1, 3, 224, 224]
            
        Returns:
            Model output [1, 1000]
        """
        # Create future for this request
        future = asyncio.Future()
        
        # Create request
        request = BatchRequest(
            request_id=str(time.time()),
            input_tensor=input_tensor,
            future=future,
            timestamp=time.time()
        )
        
        # Add to queue
        self.queue.append(request)
        self.stats['total_requests'] += 1
        
        # Wait for result
        result = await future
        return result
    
    async def _batch_processor(self):
        """Background task that processes batches"""
        while True:
            try:
                # Wait for requests or timeout
                await asyncio.sleep(0.001)  # Check every 1ms
                
                if len(self.queue) == 0:
                    continue
                
                # Check if we should process batch
                should_process = (
                    len(self.queue) >= self.max_batch_size or  # Batch full
                    (len(self.queue) > 0 and 
                     time.time() - self.queue[0].timestamp >= self.max_wait_ms)  # Timeout
                )
                
                if should_process:
                    await self._process_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processor: {e}")
    
    async def _process_batch(self):
        """Process current batch"""
        if len(self.queue) == 0:
            return
        
        # Collect batch (up to max_batch_size)
        batch_size = min(len(self.queue), self.max_batch_size)
        batch_requests = [self.queue.popleft() for _ in range(batch_size)]
        
        try:
            # Stack inputs into batch
            batch_inputs = np.concatenate([req.input_tensor for req in batch_requests], axis=0)
            
            # Run inference
            if self.use_onnx:
                # ONNX inference
                batch_outputs = self.model.predict(batch_inputs)
            else:
                # PyTorch inference
                with torch.no_grad():
                    batch_tensor = torch.from_numpy(batch_inputs)
                    batch_outputs = self.model(batch_tensor).numpy()
            
            # Distribute results to requests
            for i, request in enumerate(batch_requests):
                output = batch_outputs[i:i+1]  # Keep batch dimension
                request.future.set_result(output)
                
                # Update stats
                wait_time = time.time() - request.timestamp
                self.stats['total_wait_time'] += wait_time
            
            # Update batch stats
            self.stats['total_batches'] += 1
            self.stats['avg_batch_size'] = (
                self.stats['total_requests'] / self.stats['total_batches']
            )
            
        except Exception as e:
            # Set exception for all requests in batch
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def get_stats(self):
        """Get batching statistics"""
        avg_wait_ms = 0
        if self.stats['total_requests'] > 0:
            avg_wait_ms = (self.stats['total_wait_time'] / self.stats['total_requests']) * 1000
        
        return {
            'total_requests': self.stats['total_requests'],
            'total_batches': self.stats['total_batches'],
            'avg_batch_size': round(self.stats['avg_batch_size'], 2),
            'avg_wait_time_ms': round(avg_wait_ms, 2),
            'queue_size': len(self.queue)
        }