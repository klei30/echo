import sys
sys.path.insert(0, ".")
from memory.mem0_client import _get_memory

m = _get_memory()
uid = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"
result = m.get_all(filters={"user_id": uid})
items = result if isinstance(result, list) else result.get("results", [])
for i, item in enumerate(items):
    print(f"[{i}] id={item.get('id','?')} | {item.get('memory','')[:90]}")
