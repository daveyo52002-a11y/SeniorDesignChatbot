# Wayne State email verification setup

This is an **email ownership check**, not proof of current student enrollment. People with active @wayne.edu mailboxes (including some faculty, staff, or alumni) may be able to log in.

## Configuration on Jetstream2

1. Update your backend virtual environment:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Create `backend/.env` **on Jetstream2 only** (do not commit secrets):
   ```dotenv
   AUTH_SECRET=REPLACE_WITH_A_LONG_RANDOM_SECRET
   SMTP_HOST=smtp.your-email-provider.com
   SMTP_PORT=587
   SMTP_FROM=your-verified-sender@example.com
   SMTP_USER=your-smtp-username
   SMTP_PASSWORD=your-smtp-password
   ```
   Generate a secret: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

3. Replace the example SMTP values with credentials from an email provider that supports SMTP. The sender must be verified with that provider. Port 587 uses STARTTLS; port 465 uses SSL/TLS. Keep the SMTP password out of source control and the frontend.

4. Restart FastAPI from the backend folder:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   Restart Streamlit from the frontend folder:
   ```bash
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```
   If your deployment is using another startup method, restart the existing services with the same commands/options you already use.

## Testing

- Open Streamlit in a new/private browser session: it should show email login only.
- Try a non-Wayne email: it should be rejected.
- Enter an actual `@wayne.edu` address, then confirm the six-digit code from the received message.
- Test chat, upload, feedback, and stats after login.
- Try `curl -i http://localhost:8000/stats` from the server: it should return HTTP 401 without a bearer token.
- Click Log out, then try a previous bearer token against `/stats`: it should return HTTP 401.

## Security / deployment notes

- Protect port 8000 from public access with a firewall/bind to `127.0.0.1` when frontend and backend are on the same host. Access the Streamlit site over HTTPS (through a reverse proxy / TLS endpoint).
- For multi-instance deployments, use a shared authentication database rather than the local SQLite database.
- Sessions expire after 8 hours. Code expires in 10 minutes, with five code guesses, a 60-second resend delay, and a ten-message-per-email-per-day limit. These are basic safeguards, not global anti-abuse controls.
- For an internet-facing deployment, add IP-based rate limiting at the reverse proxy and monitor SMTP abuse/cost. Consider stronger shared session storage and key rotation.
- The `backend/auth.sqlite3` file stores hashed sessions and code digests; never commit it.
- Passwordless email sign-in does not establish student enrollment or affiliation beyond access to the mailbox.
