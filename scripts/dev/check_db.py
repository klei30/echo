import asyncio
import aiosqlite

async def check():
    async with aiosqlite.connect("echo.db") as db:
        db.row_factory = aiosqlite.Row

        # Confidence table (real name)
        async with db.execute("SELECT user_id, topic, score FROM confidence ORDER BY score DESC LIMIT 15") as cur:
            rows = await cur.fetchall()
        print("CONFIDENCE SCORES:")
        for r in rows:
            d = dict(r)
            print(f"  {d['user_id'][:20]}  topic={d['topic']}  score={d['score']:.3f}")

        # Checkpoints table
        async with db.execute("SELECT * FROM checkpoints ORDER BY rowid DESC LIMIT 10") as cur:
            rows = await cur.fetchall()
        print("\nCHECKPOINTS:")
        for r in rows:
            print(" ", dict(r))

        # Training pairs count per user
        async with db.execute("SELECT user_id, COUNT(*) as cnt FROM training_pairs GROUP BY user_id") as cur:
            rows = await cur.fetchall()
        print("\nTRAINING PAIRS COUNT:")
        for r in rows:
            d = dict(r)
            print(f"  {d['user_id'][:20]}  pairs={d['cnt']}")

asyncio.run(check())
