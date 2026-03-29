"""
Production server launcher.

Single-worker (development / profiling):
    python run.py

Multi-worker (production — each worker loads its own model instances):
    python run.py --workers 4

Workers multiply throughput near-linearly for stateless endpoints.
The DynamicBatcher is per-worker; each worker batches its own requests.
Redis cache is shared across all workers via the external Redis process.
"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastInfer server")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn worker processes (default: 1)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting FastInfer with {args.workers} worker(s) on {args.host}:{args.port}")

    uvicorn.run(
        "src.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )
