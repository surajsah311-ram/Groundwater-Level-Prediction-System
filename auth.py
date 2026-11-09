"""Optional Supabase authentication for the Streamlit dashboard.

Configure SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets to enable
real login. Without credentials, the app remains in clearly labelled demo mode.
Never store passwords in this application.
"""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


def _setting(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value.strip()
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def configured() -> bool:
    return bool(_setting("SUPABASE_URL") and _setting("SUPABASE_ANON_KEY"))


def _request(path: str, payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    base = _setting("SUPABASE_URL").rstrip("/")
    key = _setting("SUPABASE_ANON_KEY")
    try:
        r = requests.post(
            f"{base}/auth/v1/{path}",
            json=payload,
            headers={"apikey": key, "Content-Type": "application/json"},
            timeout=15,
        )
        data = r.json() if r.content else {}
        if r.ok:
            return True, "", data
        msg = data.get("msg") or data.get("message") or data.get("error_description") or "Authentication failed."
        return False, str(msg), data
    except requests.RequestException:
        return False, "Could not reach the authentication service. Check the connection and try again.", {}
    except ValueError:
        return False, "The authentication service returned an unreadable response.", {}


def require_auth() -> bool:
    """Render login/register/reset forms; return True when the app may continue."""
    if not configured():
        st.sidebar.info("Demo mode: login is not enabled because Supabase credentials are not configured.")
        return True

    if st.session_state.get("p17_user"):
        with st.sidebar:
            st.caption(f"Signed in: {st.session_state['p17_user']}")
            if st.button("Log out", use_container_width=True):
                for key in ("p17_user", "p17_access_token"):
                    st.session_state.pop(key, None)
                st.rerun()
        return True

    st.markdown("## Sign in to Groundwater Intelligence")
    st.caption("Use your account to access the dashboard. Authentication is managed by Supabase.")
    login_tab, register_tab, reset_tab = st.tabs(["Login", "Create account", "Forgot password"])
    with login_tab:
        with st.form("p17_login_form"):
            email = st.text_input("Email", key="p17_login_email")
            password = st.text_input("Password", type="password", key="p17_login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            if not email.strip() or not password:
                st.error("Enter both email and password.")
            else:
                ok, error, data = _request("token?grant_type=password", {"email": email.strip(), "password": password})
                if ok and data.get("access_token"):
                    st.session_state["p17_user"] = data.get("user", {}).get("email", email.strip())
                    st.session_state["p17_access_token"] = data["access_token"]
                    st.rerun()
                else:
                    st.error(error or "Login failed. Check your credentials.")
    with register_tab:
        with st.form("p17_register_form"):
            email = st.text_input("Email address", key="p17_register_email")
            password = st.text_input("Password (minimum 8 characters)", type="password", key="p17_register_password")
            confirm = st.text_input("Confirm password", type="password", key="p17_register_confirm")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            if not email.strip() or len(password) < 8:
                st.error("Enter an email and a password of at least 8 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, error, data = _request("signup", {"email": email.strip(), "password": password})
                if ok:
                    if data.get("access_token"):
                        st.session_state["p17_user"] = data.get("user", {}).get("email", email.strip())
                        st.session_state["p17_access_token"] = data["access_token"]
                        st.rerun()
                    else:
                        st.success("Account created. Check your email to confirm it, then log in.")
                else:
                    st.error(error or "Could not create account.")
    with reset_tab:
        with st.form("p17_reset_form"):
            email = st.text_input("Account email", key="p17_reset_email")
            submitted = st.form_submit_button("Send password reset email", use_container_width=True)
        if submitted:
            if not email.strip():
                st.error("Enter your account email.")
            else:
                ok, error, _ = _request("recover", {"email": email.strip()})
                if ok:
                    st.success("If the account exists, a password-reset email has been sent.")
                else:
                    st.error(error or "Could not request a password reset.")
    st.info("Configure Supabase in deployment secrets to enable authentication. The dashboard is hidden until login succeeds.")
    return False
