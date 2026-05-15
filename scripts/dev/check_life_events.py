import aiosqlite, asyncio

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"

async def check():
    async with aiosqlite.connect("./echo.db") as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("PRAGMA table_info(life_events)") as cur:
            cols = [r[1] for r in await cur.fetchall()]
        print(f"life_events columns: {cols}\n")

        async with db.execute(
            "SELECT event_type, event_domain, title, created_at FROM life_events WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
            (USER_ID,)
        ) as cur:
            rows = await cur.fetchall()
        print(f"=== Life events ({len(rows)}) ===")
        for r in rows:
            print(f"  [{r['created_at']}] {r['event_domain']}/{r['event_type']}: {r['title']}")

        async with db.execute("SELECT COUNT(*) as cnt FROM checkpoints WHERE user_id=?", (USER_ID,)) as cur:
            chk = (await cur.fetchone())['cnt']
        print(f"\nCheckpoints recorded: {chk}")

asyncio.run(check())
