# Retroactively seed mem0 from existing high-quality training pairs.
import asyncio
import aiosqlite
import os
import sys

# Ensure Echo's own modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.mem0_client import add_memories

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"
DB_PATH = "./echo.db"
BATCH_SIZE = 5  # memories per add() call — keeps OpenAI token cost low

async def seed():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT user_msg, assistant_msg FROM training_pairs
            WHERE user_id = ?
              AND engagement_signal != 'thumbs_down'
              AND perplexity >= 0.6
            ORDER BY created_at ASC
            LIMIT 60
            """,
            (USER_ID,),
        ) as cur:
            rows = await cur.fetchall()

    print(f"Found {len(rows)} quality pairs to seed into mem0")
    if not rows:
        print("Nothing to seed.")
        return

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        messages = []
        for r in batch:
            messages.append({"role": "user", "content": r["user_msg"]})
            messages.append({"role": "assistant", "content": r["assistant_msg"]})
        print(f"Seeding batch {i // BATCH_SIZE + 1} ({len(batch)} pairs)...")
        try:
            await add_memories(messages, user_id=USER_ID)
        except Exception as e:
            print(f"  Batch failed: {e}")

    print("\nDone. mem0 memories seeded for user.")

asyncio.run(seed())
