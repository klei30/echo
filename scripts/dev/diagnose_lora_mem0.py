# Quick diagnostic: check LoRA state in vLLM and mem0 memory count for marco.
import asyncio
import httpx
import aiosqlite
import json

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"
VLLM_URL = "http://127.0.0.1:8003/v1"
QDRANT_DB = "./qdrant_data/collection/mem0/storage.sqlite"

async def check_vllm():
    print("\n=== vLLM LoRA check ===")
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{VLLM_URL}/models")
            models = [m["id"] for m in r.json().get("data", [])]
            lora_name = f"gemma4_user_{USER_ID}"
            loaded = lora_name in models
            print(f"Models in vLLM: {models}")
            print(f"User LoRA loaded: {loaded}")
    except Exception as e:
        print(f"vLLM error: {e}")

async def check_qdrant():
    print("\n=== mem0 qdrant check ===")
    try:
        async with aiosqlite.connect(QDRANT_DB) as db:
            db.row_factory = aiosqlite.Row
            # Show schema
            async with db.execute("PRAGMA table_info(points)") as cur:
                cols = [r["name"] for r in await cur.fetchall()]
            print(f"points columns: {cols}")
            async with db.execute("SELECT COUNT(*) as cnt FROM points") as cur:
                row = await cur.fetchone()
            print(f"Total points: {row['cnt']}")
            # Try to find user-specific data
            payload_col = None
            for c in cols:
                if "payload" in c.lower() or "meta" in c.lower() or "data" in c.lower():
                    payload_col = c
                    break
            if payload_col:
                async with db.execute(
                    f"SELECT {payload_col} FROM points WHERE {payload_col} LIKE ? LIMIT 5",
                    (f'%{USER_ID}%',)
                ) as cur:
                    rows = await cur.fetchall()
                print(f"Memories for marco in '{payload_col}': {len(rows)}")
                for r in rows[:3]:
                    val = r[0]
                    try:
                        p = json.loads(val)
                        mem = p.get("memory") or p.get("text") or str(p)[:120]
                        print(f"  - {mem[:120]}")
                    except Exception:
                        print(f"  - {str(val)[:120]}")
            else:
                # Dump first row raw to see structure
                async with db.execute("SELECT * FROM points LIMIT 1") as cur:
                    row = await cur.fetchone()
                if row:
                    print(f"Sample row keys: {list(dict(row).keys())}")
                    for k, v in dict(row).items():
                        print(f"  {k}: {str(v)[:100]}")
    except Exception as e:
        print(f"Qdrant DB error: {e}")

async def check_training_pairs():
    print("\n=== Training pairs & teacher policy ===")
    try:
        async with aiosqlite.connect("./echo.db") as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?", (USER_ID,)) as cur:
                total = (await cur.fetchone())['cnt']
            async with db.execute(
                "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND perplexity>=0.6", (USER_ID,)
            ) as cur:
                quality = (await cur.fetchone())['cnt']
            async with db.execute(
                "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND engagement_signal!='thumbs_down' AND perplexity>=0.6", (USER_ID,)
            ) as cur:
                meaningful = (await cur.fetchone())['cnt']
            async with db.execute(
                "SELECT COUNT(*) as cnt FROM teacher_usage WHERE user_id=? AND created_at>=datetime('now','-1 day')", (USER_ID,)
            ) as cur:
                teacher_today = (await cur.fetchone())['cnt']
            async with db.execute(
                "SELECT COUNT(*) as cnt FROM teacher_usage WHERE user_id=? AND created_at>=datetime('now','-7 days')", (USER_ID,)
            ) as cur:
                teacher_week = (await cur.fetchone())['cnt']
            print(f"Total pairs: {total}")
            print(f"Quality pairs (perplexity>=0.6): {quality}")
            print(f"Meaningful pairs (teacher budget key): {meaningful}")
            print(f"Teacher API calls today: {teacher_today}/2  this week: {teacher_week}/5")
            if meaningful >= 20:
                print("Status: TRAINED user (mem updates only on high-importance or thumbs-up)")
            else:
                print("Status: NEW user (mem updates allowed freely, 10/day)")
    except Exception as e:
        print(f"DB error: {e}")

async def main():
    await check_vllm()
    await check_qdrant()
    await check_training_pairs()

asyncio.run(main())
