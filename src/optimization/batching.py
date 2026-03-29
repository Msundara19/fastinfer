"""
Dynamic batching for improved throughput
Collects requests and processes them in batches
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass
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
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds
        self.use_onnx = use_onnx

        self.queue = asyncio.Queue()
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0,
            'total_wait_time': 0
        }

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
        Add request to batch queue and wait for result.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = BatchRequest(
            request_id=str(time.monotonic()),
            input_tensor=input_tensor,
            future=future,
            timestamp=time.monotonic()
        )

        await self.queue.put(request)
        self.stats['total_requests'] += 1

        return await future

    async def _batch_processor(self):
        """Background task that collects and processes batches."""
        while True:
            try:
                batch = []

                # Block until the first request arrives
                first = await self.queue.get()
                batch.append(first)

                # Collect more requests up to max_batch_size within the wait window
                deadline = time.monotonic() + self.max_wait_ms
                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processor: {e}")

    def _run_inference(self, batch_inputs: np.ndarray) -> np.ndarray:
        """Synchronous inference — runs in a thread pool executor."""
        if self.use_onnx:
            return self.model.predict(batch_inputs)
        else:
            with torch.no_grad():
                batch_tensor = torch.from_numpy(batch_inputs).to(next(self.model.parameters()).device)
                output = self.model(batch_tensor)
                # MPS/CUDA tensors must be moved to CPU before numpy conversion
                return output.cpu().numpy()

    async def _process_batch(self, batch_requests: list):
        """Process a collected batch."""
        try:
            batch_inputs = np.concatenate([req.input_tensor for req in batch_requests], axis=0)

            # Run inference off the event loop so it doesn't block incoming requests
            loop = asyncio.get_event_loop()
            batch_outputs = await loop.run_in_executor(None, self._run_inference, batch_inputs)

            for i, request in enumerate(batch_requests):
                request.future.set_result(batch_outputs[i:i+1])
                self.stats['total_wait_time'] += time.monotonic() - request.timestamp

            self.stats['total_batches'] += 1
            self.stats['avg_batch_size'] = (
                self.stats['total_requests'] / self.stats['total_batches']
            )

        except Exception as e:
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
            'queue_size': self.queue.qsize()
        }