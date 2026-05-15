# Check mem0 memories for marco using the mem0 library directly.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.mem0_client import _get_memory

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"

m = _get_memory()
result = m.get_all(filters={"user_id": USER_ID}, top_k=100)
items = result if isinstance(result, list) else result.get("results", [])
print(f"Memories for marco: {len(items)}")
for i, item in enumerate(items[:20]):
    mem = item.get("memory") or item.get("text") or str(item)[:100]
    print(f"  [{i+1}] {mem}")
