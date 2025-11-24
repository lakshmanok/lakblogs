import redis
import os

# Assuming Redis is running locally on default port
# In a real app, we'd use env vars
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping() # Check connection
    print("Connected to Redis")
except redis.ConnectionError:
    print("Could not connect to Redis. Caching will be disabled/fail.")
    r = None

def get_cached_cell(ticker: str, column: str) -> str | None:
    if not r: return None
    key = f"stock_grid:{ticker}:{column}"
    return r.get(key)

def set_cached_cell(ticker: str, column: str, value: str):
    if not r: return
    key = f"stock_grid:{ticker}:{column}"
    r.set(key, value)
