"""
Redis-backed prediction cache.

Cache key  : MD5 hash of raw image bytes
Cache value: JSON-serialised prediction result
TTL        : configurable via CACHE_TTL (default 1 hour)

Caching eliminates preprocessing + inference (~8.5ms + 1.33ms) entirely
for repeated inputs, reducing latency to ~0.3ms (Redis round-trip).
"""

import hashlib
import json
import redis.asyncio as aioredis
from typing import Optional


class PredictionCache:
    def __init__(self, host: str, port: int, ttl: int):
        self.ttl = ttl
        self._client: Optional[aioredis.Redis] = None
        self._host = host
        self._port = port
        self.hits = 0
        self.misses = 0

    async def connect(self):
        self._client = aioredis.Redis(
            host=self._host,
            port=self._port,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        await self._client.ping()
        print(f"Redis cache connected ({self._host}:{self._port}, TTL={self.ttl}s)")

    async def close(self):
        if self._client:
            await self._client.aclose()

    def _key(self, image_bytes: bytes) -> str:
        return "fastinfer:" + hashlib.md5(image_bytes).hexdigest()

    async def get(self, image_bytes: bytes) -> Optional[dict]:
        if not self._client:
            return None
        try:
            value = await self._client.get(self._key(image_bytes))
            if value:
                self.hits += 1
                result = json.loads(value)
                result["cache"] = "hit"
                return result
        except Exception:
            pass
        self.misses += 1
        return None

    async def set(self, image_bytes: bytes, result: dict):
        if not self._client:
            return
        try:
            payload = {k: v for k, v in result.items() if k != "cache"}
            await self._client.setex(
                self._key(image_bytes),
                self.ttl,
                json.dumps(payload),
            )
        except Exception:
            pass

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
            "ttl_seconds": self.ttl,
        }
