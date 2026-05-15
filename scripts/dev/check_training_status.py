import aiosqlite, asyncio

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"

async def check():
    async with aiosqlite.connect("./echo.db") as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT * FROM training_runs WHERE user_id=? ORDER BY started_at DESC LIMIT 5",
            (USER_ID,)
        ) as cur:
            rows = await cur.fetchall()
        print("=== Training runs (latest 5) ===")
        if not rows:
            print("  None found")
        for r in rows:
            d = dict(r)
            print(f"  id={d.get('id')} status={d.get('status')} lane={d.get('lane')}")
            print(f"    started={d.get('started_at')} finished={d.get('finished_at','n/a')}")
            print(f"    pairs={d.get('untrained_pairs')} error={d.get('error')}")
            print(f"    adapter={str(d.get('adapter_path',''))[:80]}")

        # Also check WSL vLLM processes via netstat to see if training is happening
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND used_in_training=0 AND perplexity>=0.6 AND engagement_signal!='thumbs_down'",
            (USER_ID,)
        ) as cur:
            row = await cur.fetchone()
        print(f"\nUntrained pairs available: {row['cnt']}")

asyncio.run(check())
