import sqlite3
import hashlib
import secrets
import json
import os
from crypto_utils import encrypt_value, decrypt_value
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "creditflow.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('msme', 'officer'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msme_username TEXT NOT NULL,
            raw_text TEXT,
            numbers_found TEXT,
            submitted_at TEXT,
            status TEXT DEFAULT 'pending',
            daily_revenue REAL, restock_frequency REAL, restock_amount REAL,
            pos_sales_consistency REAL, supplier_payment_delay REAL,
            revenue_volatility REAL, months_operating REAL,
            score INTEGER, risk TEXT, max_loan INTEGER,
            scored_by TEXT, scored_at TEXT,
            requested_amount REAL, requested_at TEXT,
            approved_by TEXT, approved_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            username TEXT PRIMARY KEY,
            full_name TEXT,
            venture_name TEXT,
            nin TEXT,
            photo_path TEXT,
            biometric_path TEXT,
            account_number TEXT,
            gender TEXT,
            date_of_birth TEXT,
            email TEXT,
            phone_number TEXT,
            address TEXT,
            updated_at TEXT
        )
    """)
    for col_def in ["phone_verified INTEGER DEFAULT 0"]:
        try:
            conn.execute(f"ALTER TABLE profiles ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    for col_def in ["requested_amount REAL", "requested_at TEXT", "approved_by TEXT", "approved_at TEXT"]:
        try:
            conn.execute(f"ALTER TABLE submissions ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()
    
# ---- Passwords ----
def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000).hex()
    return pw_hash, salt

def create_user(username, password, role):
    conn = get_connection()
    try:
        pw_hash, salt = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
            (username, pw_hash, salt, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    row = conn.execute("SELECT password_hash, salt, role FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row is None:
        return None
    check_hash, _ = hash_password(password, row["salt"])
    return row["role"] if check_hash == row["password_hash"] else None

# ---- Sessions (survive browser refresh via URL token) ----
def create_session(username, role):
    token = secrets.token_urlsafe(24)
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (token, username, role, created_at) VALUES (?, ?, ?, ?)",
        (token, username, role, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return token

def get_session(token):
    conn = get_connection()
    row = conn.execute("SELECT username, role FROM sessions WHERE token = ?", (token,)).fetchone()
    conn.close()
    return row

def delete_session(token):
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()

# ---- Submissions ----
def create_submission(msme_username, raw_text, numbers_found):
    conn = get_connection()
    conn.execute(
        "INSERT INTO submissions (msme_username, raw_text, numbers_found, submitted_at, status) VALUES (?, ?, ?, ?, 'pending')",
        (msme_username, raw_text, json.dumps(numbers_found), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_pending_submissions():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM submissions WHERE status = 'pending' ORDER BY submitted_at DESC").fetchall()
    conn.close()
    return rows

def get_submission(sub_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM submissions WHERE id = ?", (sub_id,)).fetchone()
    conn.close()
    return row

def get_my_submissions(msme_username):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM submissions WHERE msme_username = ? ORDER BY submitted_at DESC", (msme_username,)
    ).fetchall()
    conn.close()
    return rows

def save_score(sub_id, features, score_val, risk, max_loan, officer_username):
    conn = get_connection()
    conn.execute("""
        UPDATE submissions SET
            daily_revenue=?, restock_frequency=?, restock_amount=?, pos_sales_consistency=?,
            supplier_payment_delay=?, revenue_volatility=?, months_operating=?,
            score=?, risk=?, max_loan=?, scored_by=?, scored_at=?, status='scored'
        WHERE id=?
    """, (
        features["daily_revenue"], features["restock_frequency"], features["restock_amount"],
        features["pos_sales_consistency"], features["supplier_payment_delay"],
        features["revenue_volatility"], features["months_operating"],
        score_val, risk, max_loan, officer_username, datetime.now().isoformat(), sub_id
    ))
    conn.commit()
    conn.close()

def get_all_scored_submissions():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM submissions WHERE score IS NOT NULL ORDER BY score DESC").fetchall()
    conn.close()
    return rows

# ---- Loan requests ----
def request_loan(sub_id, amount):
    conn = get_connection()
    conn.execute(
        "UPDATE submissions SET status='requested', requested_amount=?, requested_at=? WHERE id=?",
        (amount, datetime.now().isoformat(), sub_id)
    )
    conn.commit()
    conn.close()

def get_loan_requests():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM submissions WHERE status='requested' ORDER BY requested_at ASC").fetchall()
    conn.close()
    return rows

def approve_loan(sub_id, officer_username):
    conn = get_connection()
    conn.execute(
        "UPDATE submissions SET status='approved', approved_by=?, approved_at=? WHERE id=?",
        (officer_username, datetime.now().isoformat(), sub_id)
    )
    conn.commit()
    conn.close()

def reject_loan(sub_id, officer_username):
    conn = get_connection()
    conn.execute(
        "UPDATE submissions SET status='rejected', approved_by=?, approved_at=? WHERE id=?",
        (officer_username, datetime.now().isoformat(), sub_id)
    )
    conn.commit()
    conn.close()

def save_profile(username, data, photo_path=None, biometric_path=None):
    conn = get_connection()
    existing = conn.execute("SELECT photo_path, biometric_path FROM profiles WHERE username=?", (username,)).fetchone()
    final_photo = photo_path if photo_path else (existing["photo_path"] if existing else None)
    final_biometric = biometric_path if biometric_path else (existing["biometric_path"] if existing else None)
    conn.execute("""
        INSERT INTO profiles (username, full_name, venture_name, nin, photo_path, biometric_path,
            account_number, gender, date_of_birth, email, phone_number, address, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            full_name=excluded.full_name, venture_name=excluded.venture_name, nin=excluded.nin,
            photo_path=excluded.photo_path, biometric_path=excluded.biometric_path,
            account_number=excluded.account_number, gender=excluded.gender,
            date_of_birth=excluded.date_of_birth, email=excluded.email,
            phone_number=excluded.phone_number, address=excluded.address, updated_at=excluded.updated_at
    """, (
        username, data["full_name"], data["venture_name"], encrypt_value(data["nin"]), final_photo, final_biometric,
        encrypt_value(data["account_number"]), data["gender"], encrypt_value(data["date_of_birth"]),
        encrypt_value(data["email"]), encrypt_value(data["phone_number"]), encrypt_value(data["address"]),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def get_profile(username):
    conn = get_connection()
    row = conn.execute("SELECT * FROM profiles WHERE username=?", (username,)).fetchone()
    conn.close()
    if row is None:
        return None
    profile = dict(row)
    for field in ["nin", "account_number", "date_of_birth", "email", "phone_number", "address"]:
        profile[field] = decrypt_value(profile[field])
    return profile

def get_officer_visible_profile(username):
    conn = get_connection()
    row = conn.execute(
        "SELECT full_name, venture_name, photo_path, account_number, gender FROM profiles WHERE username=?",
        (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    profile = dict(row)
    profile["account_number"] = decrypt_value(profile["account_number"])
    return profile

def update_password(username, current_password, new_password):
    if verify_user(username, current_password) is None:
        return False, "Current password is incorrect."
    if not new_password:
        return False, "New password cannot be empty."
    pw_hash, salt = hash_password(new_password)
    conn = get_connection()
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?", (pw_hash, salt, username))
    conn.commit()
    conn.close()
    return True, "Password updated."