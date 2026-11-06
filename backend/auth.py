"""Email one-time codes and bearer sessions for Wayne State email holders."""
import hashlib
import hmac
import os
import secrets
import smtplib
import sqlite3
import ssl
import time
from email.message import EmailMessage
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

DB = Path(__file__).parent / "auth.sqlite3"
bearer = HTTPBearer(auto_error=False)
CODE_TTL = 600  # 10 minutes
SESSION_TTL = 8 * 60 * 60
MAX_ATTEMPTS = 5

class EmailRequest(BaseModel):
    email: EmailStr

class VerifyRequest(EmailRequest):
    code: str

def normalize(email):
    email = str(email).strip().lower()
    if not email.endswith("@wayne.edu") or email.count("@") != 1:
        raise HTTPException(400, "Please use a @wayne.edu email address.")
    return email

def connection():
    con = sqlite3.connect(DB, timeout=15)
    con.execute("""CREATE TABLE IF NOT EXISTS codes
        (email TEXT PRIMARY KEY, digest TEXT NOT NULL, expires INTEGER NOT NULL,
         attempts INTEGER NOT NULL, last_sent INTEGER NOT NULL, sent_count INTEGER NOT NULL,
         day INTEGER NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS sessions
        (token_digest TEXT PRIMARY KEY, email TEXT NOT NULL, expires INTEGER NOT NULL)""")
    return con

def secret_key():
    secret = os.getenv("AUTH_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(503, "Email login is not configured (AUTH_SECRET).")
    return secret.encode()

def code_digest(email, code):
    return hmac.new(secret_key(), (email + ":" + code).encode(), hashlib.sha256).hexdigest()

def send_email(email, code):
    host = os.getenv("SMTP_HOST")
    sender = os.getenv("SMTP_FROM")
    if not host or not sender:
        raise HTTPException(503, "Email login is not configured (SMTP_HOST/SMTP_FROM).")
    message = EmailMessage()
    message["Subject"] = "AI Laboratory Assistant verification code"
    message["From"] = sender
    message["To"] = email
    message.set_content("Your AI Laboratory Assistant verification code is: " + code +
                        "\nThis code expires in 10 minutes. Do not share it.")
    port = int(os.getenv("SMTP_PORT", "587"))
    context = ssl.create_default_context()
    try:
        if port == 465:
            smtp = smtplib.SMTP_SSL(host, port, timeout=15, context=context)
        else:
            smtp = smtplib.SMTP(host, port, timeout=15)
        with smtp:
            if port != 465:
                smtp.starttls(context=context)
            user = os.getenv("SMTP_USER")
            password = os.getenv("SMTP_PASSWORD")
            if user and password:
                smtp.login(user, password)
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise HTTPException(503, "Could not send verification email. Try again later.") from exc

def request_code(email):
    email = normalize(email)
    now = int(time.time())
    day = now // 86400
    with connection() as con:
        row = con.execute("SELECT last_sent, sent_count, day FROM codes WHERE email=?", (email,)).fetchone()
        if row and now - row[0] < 60:
            raise HTTPException(429, "Please wait 60 seconds before requesting another code.")
        if row and row[2] == day and row[1] >= 10:
            raise HTTPException(429, "Daily verification limit reached.")
        code = f"{secrets.randbelow(1000000):06d}"
        digest = code_digest(email, code)
        count = (row[1] + 1) if row and row[2] == day else 1
        # Send before storing the code: a failed SMTP attempt must not invalidate a working code.
        send_email(email, code)
        con.execute("""INSERT INTO codes(email,digest,expires,attempts,last_sent,sent_count,day)
            VALUES(?,?,?,?,?,?,?) ON CONFLICT(email) DO UPDATE SET
            digest=excluded.digest, expires=excluded.expires, attempts=0,
            last_sent=excluded.last_sent, sent_count=excluded.sent_count, day=excluded.day""",
            (email, digest, now + CODE_TTL, 0, now, count, day))
    return {"message": "Verification code sent. Check your Wayne State email."}

def verify_code(email, code):
    email = normalize(email)
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        raise HTTPException(400, "Enter the six-digit code.")
    now = int(time.time())
    with connection() as con:
        row = con.execute("SELECT digest, expires, attempts FROM codes WHERE email=?", (email,)).fetchone()
        if not row or now > row[1] or row[2] >= MAX_ATTEMPTS:
            raise HTTPException(401, "Code expired or attempts exceeded. Request a new code.")
        # Count incorrect guesses even when the code is invalid.
        if not hmac.compare_digest(row[0], code_digest(email, code)):
            con.execute("UPDATE codes SET attempts=attempts+1 WHERE email=?", (email,))
            raise HTTPException(401, "Incorrect verification code.")
        con.execute("DELETE FROM codes WHERE email=?", (email,))
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        con.execute("DELETE FROM sessions WHERE expires < ?", (now,))
        con.execute("INSERT INTO sessions(token_digest,email,expires) VALUES(?,?,?)",
                    (token_hash, email, now + SESSION_TTL))
    return {"access_token": token, "token_type": "bearer", "expires_in": SESSION_TTL}

def authenticated_email(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Please log in with your Wayne State email.")
    digest = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    with connection() as con:
        row = con.execute("SELECT email, expires FROM sessions WHERE token_digest=?", (digest,)).fetchone()
    if not row or row[1] <= int(time.time()):
        raise HTTPException(401, "Login expired. Please sign in again.")
    return row[0]

def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if credentials:
        digest = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        with connection() as con:
            con.execute("DELETE FROM sessions WHERE token_digest=?", (digest,))
    return {"logged_out": True}
