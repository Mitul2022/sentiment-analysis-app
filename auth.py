import streamlit as st
import bcrypt
import pyotp
import qrcode
import io
from db_connection import get_connection

# ------------------- DB UTILS ------------------- #
def run_query(query, params=None, fetch=False, many=False):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                if params is not None and not isinstance(params, (list, tuple)):
                    params = (params,)
                if many:
                    cur.executemany(query, params)
                else:
                    cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
    finally:
        conn.close()

def get_user(username):
    rows = run_query(
        "SELECT user_id, username, password_hash, totp_secret FROM users WHERE username = %s;",
        (username,), fetch=True
    )
    if not rows:
        return None
    user = rows[0]
    if isinstance(user, dict):
        return user
    else:
        user_id, uname, pwd_hash, totp_secret = user
        return {
            "user_id": user_id,
            "username": uname,
            "password_hash": pwd_hash,
            "totp_secret": totp_secret
        }

def update_totp_secret(user_id, secret):
    run_query("UPDATE users SET totp_secret = %s WHERE user_id = %s;", (secret, user_id))

# ------------------- Registration ------------------- #
def registration_page():
    st.subheader("📝 Register New User")

    new_username = st.text_input("Choose Username")
    new_password = st.text_input("Choose Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if not new_username or not new_password:
            st.error("Username and password required.")
            return
        if new_password != confirm_password:
            st.error("Passwords do not match.")
            return
        exists = run_query("SELECT 1 FROM users WHERE username = %s;", (new_username,), fetch=True)
        if exists:
            st.error("Username already exists.")
            return
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        secret = pyotp.random_base32()
        run_query("INSERT INTO users (username, password_hash, totp_secret) VALUES (%s, %s, %s);",
                  (new_username, hashed, secret))

        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(new_username, issuer_name="SentimentApp")
        qr_img = qrcode.make(totp_uri)
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")

        st.success("Registered! Scan this QR code in Google Authenticator.")
        st.image(buf.getvalue())

# ------------------- Login Page ------------------- #
def login_page():
    st.subheader("🔑 User Login")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        user = get_user(username)
        if not user:
            st.error("Invalid username.")
            return
        if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            st.error("Invalid password.")
            return
        st.session_state["awaiting_otp"] = True
        st.session_state["auth_user"] = user
        st.success("Password correct. Enter your OTP below.")

    if st.session_state.get("awaiting_otp"):
        otp = st.text_input("Enter 6-digit OTP", max_chars=6, key="otp_input")
        if st.button("Verify OTP"):
            user = st.session_state["auth_user"]
            totp = pyotp.TOTP(user["totp_secret"])
            if totp.verify(otp):
                st.session_state["authenticated"] = True
                st.session_state["awaiting_otp"] = False
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid OTP")

# ------------------- Logout ------------------- #
def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.session_state["auth_user"] = None
        st.session_state["awaiting_otp"] = False
        st.rerun()

# ------------------- Auth Controller ------------------- #
def auth_controller():
    if st.session_state.get("authenticated", False):
        user = st.session_state.get("auth_user")
        st.sidebar.markdown(f"👋 Hello, **{user['username']}**")
        logout_button()
        return True
    else:
        st.sidebar.title("User Access")
        page = st.sidebar.radio("Select:", ["Login", "Register"])
        if page == "Register":
            registration_page()
        else:
            login_page()
        if not st.session_state.get("authenticated", False):
            st.stop()
        return True
