import sqlite3, json
from pathlib import Path

db = Path("models/logs.sqlite")

if not db.exists():
    print("DB not found:", db)
    raise SystemExit(1)

conn = sqlite3.connect(db)
cur = conn.cursor()

print("\nTotal rows in logs:")
cur.execute("SELECT count(*) FROM predictions")
print(cur.fetchone()[0])

print("\nLast 5 rows:")
cur.execute("""
SELECT id, ts, pred_prob, risk_bucket, recommended_action, input_json 
FROM predictions 
ORDER BY id DESC 
LIMIT 5
""")

rows = cur.fetchall()

for r in rows:
    print("----------------------------------")
    print("ID:", r[0])
    print("Timestamp:", r[1])
    print("Pred Prob:", r[2])
    print("Bucket:", r[3])
    print("Action:", r[4])
    print("Input JSON:", r[5])
    print("----------------------------------")

conn.close()