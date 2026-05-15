import asyncio
import aiosqlite
import json

async def check_memories():
    async with aiosqlite.connect("qdrant_data/collection/mem0/storage.sqlite") as db:
        db.row_factory = aiosqlite.Row
        
        # List tables
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
            tables = await cur.fetchall()
            print("Tables:", [t[0] for t in tables])
        
        # Get user memory counts
        async with db.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """) as cur:
            tables = await cur.fetchall()
            
        for table in tables:
            table_name = table[0]
            print(f"\n=== Table: {table_name} ===")
            try:
                async with db.execute(f"SELECT * FROM {table_name} LIMIT 5") as cur:
                    rows = await cur.fetchall()
                    for r in rows:
                        d = dict(r)
                        print(f"  {json.dumps(d, default=str)[:200]}")
            except Exception as e:
                print(f"  Error: {e}")

asyncio.run(check_memories())