import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.adapter import hot_swap_adapter, adapter_path_for_user, adapter_status

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"
LANE = "gemma4_e2b"

async def main():
    path = adapter_path_for_user(USER_ID, lane=LANE)
    print(f"Adapter path: {path}")
    if not path:
        print("No adapter found on disk.")
        return
    status = await adapter_status(USER_ID, lane=LANE)
    print(f"Currently loaded in vLLM: {status['loaded']} (serving: {status['serving_model']})")
    print(f"Loading adapter...")
    ok = await hot_swap_adapter(USER_ID, path, record_checkpoint=True, lane=LANE)
    print(f"Hot-swap result: {'SUCCESS' if ok else 'FAILED'}")
    status2 = await adapter_status(USER_ID, lane=LANE)
    print(f"Now loaded: {status2['loaded']} (serving: {status2['serving_model']})")

asyncio.run(main())
