import aiosqlite, asyncio

USER_ID = "0abcba6b-2a4a-4a66-951c-e5e6a68f1da3"

async def check():
    async with aiosqlite.connect("./echo.db") as db:
        db.row_factory = aiosqlite.Row

        # Check actual columns
        async with db.execute("PRAGMA table_info(training_pairs)") as cur:
            cols = [r[1] for r in await cur.fetchall()]
        print(f"training_pairs columns: {cols}\n")

        async with db.execute(
            "SELECT engagement_signal, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY engagement_signal ORDER BY cnt DESC",
            (USER_ID,)
        ) as cur:
            rows = await cur.fetchall()
        print("=== Engagement signals ===")
        for r in rows:
            print(f"  {r['engagement_signal']}: {r['cnt']}")

        async with db.execute(
            "SELECT engagement_signal, perplexity, topic, created_at, substr(user_msg,1,60) as msg FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
            (USER_ID,)
        ) as cur:
            rows = await cur.fetchall()
        print("\n=== Last 10 pairs ===")
        for r in rows:
            print(f"  [{r['created_at']}] signal={r['engagement_signal']} perp={r['perplexity']} topic={r['topic']}")
            print(f"    {r['msg']}")

        async with db.execute(
            "SELECT outcome, score, created_at FROM shadow_outcomes WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
            (USER_ID,)
        ) as cur:
            rows = await cur.fetchall()
        print("\n=== Recent shadow outcomes ===")
        if not rows:
            print("  None recorded yet")
        for r in rows:
            print(f"  [{r['created_at']}] outcome={r['outcome']} score={r['score']}")

        # Count untrained quality pairs using correct column name
        used_col = "used_in_training" if "used_in_training" in cols else None
        if used_col:
            async with db.execute(
                f"SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND {used_col}=0 AND perplexity>=0.6 AND engagement_signal!='thumbs_down'",
                (USER_ID,)
            ) as cur:
                row = await cur.fetchone()
            print(f"\n=== Untrained quality pairs ready for next run: {row['cnt']} (need 20) ===")
        else:
            # No used_in_training col — count all quality pairs
            async with db.execute(
                "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND perplexity>=0.6 AND engagement_signal!='thumbs_down'",
                (USER_ID,)
            ) as cur:
                row = await cur.fetchone()
            print(f"\n=== Quality pairs total: {row['cnt']} (need 20 untrained to trigger) ===")

asyncio.run(check())
