# ui.py
# Streamlit UI for AgenTest — calls the FastAPI backend over HTTP.
#
# Why HTTP instead of direct class calls?
# - Decouples UI and backend lifecycles
# - Lets you run backend separately (and swap models/providers later)
# - Aligns with future Controller integration

from __future__ import annotations

import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # read .env if present

# Config
SERVICE_BASE = os.getenv("LLM_SERVICE_BASE", "http://127.0.0.1:19090")

st.set_page_config(page_title="AgenTest - LLM UI", layout="wide")
st.title("AgenTest – LLM UI")
st.caption("Simple runner: enter a single test step, execute, review logs & result.")

# Session state
if "logs" not in st.session_state:
    st.session_state.logs = []
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False

def reset():
    st.session_state.logs = []
    st.session_state.result = None

# Health
with st.expander("Service Health", expanded=False):
    try:
        h = requests.get(f"{SERVICE_BASE}/healthz", timeout=5).json()
        st.json(h)
        if not h.get("has_api_key"):
            st.warning("OPENROUTER_API_KEY is missing on backend.")
    except Exception as e:
        st.error(f"Health check failed: {e}")

st.divider()
st.subheader("New Test Step")

step = st.text_area("Step to Execute", height=100,
                    placeholder="e.g., Click the Settings button in the toolbar")
expected = st.text_area("Expected Result", height=100,
                        placeholder="e.g., Settings window should open")
note = st.text_area("Note to LLM (optional)", height=80,
                    placeholder="e.g., If multiple Settings buttons exist, pick the main one")

c1, c2, c3 = st.columns([1.2, 1.2, 6])
with c1:
    run_clicked = st.button("▶️ Run Step", use_container_width=True, type="primary", disabled=st.session_state.running)
with c2:
    clear_clicked = st.button("🔄 Clear", use_container_width=True)

if clear_clicked:
    reset()
    st.rerun()

if run_clicked:
    if not step.strip():
        st.warning("Please enter a Step to Execute.")
    elif not expected.strip():
        st.warning("Please enter an Expected Result.")
    else:
        st.session_state.running = True
        reset()
        with st.spinner("Executing via LLM backend..."):
            try:
                payload = {"step": step, "expected": expected, "note": note}
                r = requests.post(f"{SERVICE_BASE}/run_step", json=payload, timeout=120)
                r.raise_for_status()
                body = r.json()
                st.session_state.logs = body.get("logs", [])
                st.session_state.result = body
            except Exception as e:
                st.error(f"Run failed: {e}")
            finally:
                st.session_state.running = False
        st.rerun()

st.divider()
st.subheader("Execution Log")
if st.session_state.logs:
    st.code("\n".join(st.session_state.logs), language="text")
else:
    st.info("No logs yet.")

if st.session_state.result:
    st.divider()
    st.subheader("Result")
    res = st.session_state.result
    if res.get("passed"):
        st.success("✅ TEST PASSED")
    else:
        st.error("❌ TEST FAILED")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Step**")
        st.info(res.get("step"))
        st.markdown("**Expected**")
        st.info(res.get("expected"))
    with cr:
        st.markdown("**Actual**")
        actual = res.get("actual") or ""
        (st.success if res.get("passed") else st.error)(actual)

    if res.get("verification"):
        with st.expander("Verification Details"):
            st.json(res["verification"])
    if res.get("action_plan"):
        with st.expander("Action Plan (LLM)"):
            st.json(res["action_plan"])

st.divider()
st.caption(f"LLM Service: {SERVICE_BASE}")