import sqlite3
db = sqlite3.connect('echo.db')
db.row_factory = sqlite3.Row
uid = '0abcba6b-2a4a-4a66-951c-e5e6a68f1da3'

print('=== TRAINING PAIR ANALYSIS (Marc) ===')

# What get_sft_pairs would return
qual = db.execute('''
    SELECT COUNT(*) as cnt, topic, model_used, engagement_signal, perplexity
    FROM training_pairs
    WHERE user_id=? AND used_in_training=0 AND perplexity>=0.6 AND model_used!="local"
    GROUP BY topic
''', (uid,)).fetchall()
print('\nQualifying pairs for SFT (unused, quality>=0.6, NOT local):')
total_sft = 0
for row in qual:
    print(f'  {row[1]}: {row[0]} pairs | model: {row[2]}')
    total_sft += row[0]
print(f'TOTAL: {total_sft}')

# What get_dpo_pairs would return
dpo = db.execute('''
    SELECT engagement_signal, COUNT(*) as cnt
    FROM training_pairs
    WHERE user_id=? AND used_in_training=0 AND engagement_signal IN ("thumbs_up","thumbs_down")
    GROUP BY engagement_signal
''', (uid,)).fetchall()
print('\nDPO pairs (thumbs_up/down, unused):')
for row in dpo:
    print(f'  {row[0]}: {row[1]}')

# Check if DPO can form (needs both up and down for same prompt)
up_pairs = db.execute('''
    SELECT user_msg, assistant_msg
    FROM training_pairs
    WHERE user_id=? AND engagement_signal="thumbs_up"
    ORDER BY rowid DESC LIMIT 10
''', (uid,)).fetchall()
down_pairs = db.execute('''
    SELECT user_msg, assistant_msg
    FROM training_pairs
    WHERE user_id=? AND engagement_signal="thumbs_down"
    ORDER BY rowid DESC LIMIT 10
''', (uid,)).fetchall()
print(f'\nThumbs up pairs: {len(up_pairs)}')
for p in up_pairs:
    print(f'  Q: {p[0][:60]}')
print(f'\nThumbs down pairs: {len(down_pairs)}')
for p in down_pairs:
    print(f'  Q: {p[0][:60]}')

# Overlap check
up_msgs = {p[0] for p in up_pairs}
down_msgs = {p[0] for p in down_pairs}
overlap = up_msgs & down_msgs
print(f'\nMessages with BOTH thumbs_up AND thumbs_down: {len(overlap)}')
for m in overlap:
    print(f'  {m[:60]}')

# All untrained pairs
all_unused = db.execute('''
    SELECT COUNT(*) as cnt, model_used, perplexity FROM training_pairs
    WHERE user_id=? AND used_in_training=0
    GROUP BY model_used
''', (uid,)).fetchall()
print('\nAll untrained pairs by model:')
for row in all_unused:
    print(f'  {row[1]}: {row[0]} pairs, avg perplexity: {row[2]:.2f}')

# Quality threshold check
qt_check = db.execute('''
    SELECT COUNT(*) as cnt FROM training_pairs
    WHERE user_id=? AND used_in_training=0 AND perplexity>=0.6
''', (uid,)).fetchone()
print(f'\nPairs with quality>=0.6: {qt_check[0]}')

# Local model check
local_check = db.execute('''
    SELECT COUNT(*) as cnt FROM training_pairs
    WHERE user_id=? AND used_in_training=0 AND model_used="local"
''', (uid,)).fetchone()
non_local = db.execute('''
    SELECT COUNT(*) as cnt FROM training_pairs
    WHERE user_id=? AND used_in_training=0 AND model_used!="local" AND perplexity>=0.6
''', (uid,)).fetchone()
print(f'Unused local pairs: {local_check[0]}')
print(f'Unused non-local with quality>=0.6: {non_local[0]}')