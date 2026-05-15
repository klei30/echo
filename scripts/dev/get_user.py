import sqlite3
conn = sqlite3.connect('C:/Users/ASUS/Desktop/echo/echo.db')
cur = conn.cursor()
cur.execute("SELECT id, email FROM users WHERE email='marco@gmail.com'")
print(cur.fetchone())