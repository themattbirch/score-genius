# backend/caching/redis_config.py
import os
import redis

def get_redis_client():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)
    client = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True
    )
    return client

if __name__ == "__main__":
    client = get_redis_client()
    print("Connected to Redis:", client.ping())
