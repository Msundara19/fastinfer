from locust import HttpUser, task, between
import os
import random

class FastInferUser(HttpUser):
    """
    Simulates users making predictions to FastInfer API
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Check if server is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            print("WARNING: Server health check failed!")
    
    @task(10)  # Weight: 10 (most common task)
    def predict_image(self):
        """Make prediction on test image"""
        # In production, you'd load actual test images
        # For now, we'll test the endpoint exists
        files = {'file': ('test.jpg', open('test_dog.jpg', 'rb'), 'image/jpeg')}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    latency = json_data.get('latency_ms', 0)
                    
                    # Track if latency is acceptable
                    if latency > 500:
                        response.failure(f"Latency too high: {latency}ms")
                    else:
                        response.success()
                except Exception as e:
                    response.failure(f"Failed to parse response: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)  # Weight: 2 (less common)
    def health_check(self):
        """Check server health"""
        self.client.get("/health")
    
    @task(1)  # Weight: 1 (least common)
    def get_model_info(self):
        """Get model information"""
        self.client.get("/model/info")