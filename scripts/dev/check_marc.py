import sqlite3
db = sqlite3.connect('echo.db')
db.row_factory = sqlite3.Row

uid = '0abcba6b-2a4a-4a66-951c-e5e6a68f1da3'
uid2 = 'c18c8e8c-737b-4775-b79c-4f3ebd177412'

print('=== ALL TABLES ===')
tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print([t[0] for t in tables])

print('\n=== MARC PROFILE ===')
u = db.execute('SELECT id, email, username, created_at FROM users WHERE id=?', (uid,)).fetchone()
print(dict(u))

print('\n=== MARC TOPIC HISTORY ===')
th = db.execute('SELECT topic, week_number, count FROM topic_history WHERE user_id=?', (uid,)).fetchall()
for t in th:
    print(dict(t))

print('\n=== BEN TOPIC HISTORY ===')
th = db.execute('SELECT topic, week_number, count FROM topic_history WHERE user_id=?', (uid2,)).fetchall()
for t in th:
    print(dict(t))

print('\n=== MARC SKILLS ===')
skills = db.execute('SELECT skill_name, trigger, procedure FROM user_skills WHERE user_id=? AND active=1', (uid,)).fetchall()
print(f'{len(skills)} skills')
for s in skills:
    print('-', dict(s))

print('\n=== MARC RULES ===')
rules = db.execute('SELECT rule_text, applies_to, confidence FROM user_rules WHERE user_id=? AND active=1', (uid,)).fetchall()
print(f'{len(rules)} rules')
for r in rules:
    print('-', dict(r))

print('\n=== MARC FIRST PAIR ===')
first = db.execute('SELECT user_msg, topic, created_at FROM training_pairs WHERE user_id=? ORDER BY created_at ASC LIMIT 3', (uid,)).fetchall()
for f in first:
    print(dict(f))

print('\n=== MARC PAIR TOPICS ===')
topics = db.execute('SELECT topic, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY topic', (uid,)).fetchall()
for t in topics:
    print(dict(t))

print('\n=== MARC USED IN TRAINING ===')
used = db.execute('SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND used_in_training=1', (uid,)).fetchone()
unused = db.execute('SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND used_in_training=0', (uid,)).fetchone()
print('Used:', used[0], 'Unused:', unused[0])

print('\n=== MARC MODEL USAGE ===')
models = db.execute('SELECT model_used, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY model_used', (uid,)).fetchall()
for m in models:
    print(dict(m))

print('\n=== MARC ENGAGEMENT ===')
eng = db.execute('SELECT engagement_signal, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY engagement_signal', (uid,)).fetchall()
for e in eng:
    print(dict(e))