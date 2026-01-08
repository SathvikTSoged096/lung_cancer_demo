import sqlite3
import bcrypt

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            name TEXT,
            email TEXT,
            password_hash BLOB,
            role TEXT
        )
    """)
    conn.commit()
    conn.close()

def create_user(username, name, email, password, role="user"):
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (username, name, email, password_hash, role) VALUES (?, ?, ?, ?, ?)",
        (username, name, email, password_hash, role)
    )
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT password_hash, name, role FROM users WHERE username=?",
        (username,)
    )
    row = c.fetchone()
    conn.close()

    if row and bcrypt.checkpw(password.encode(), row[0]):
        return True, row[1], row[2]
    return False, None, None

