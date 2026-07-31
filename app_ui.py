import os

import pandas as pd
import requests
import streamlit as st

# Defaults to the local server. Point at a deployed instance with:
#   MEDIDATA_API=http://<host>:8000 streamlit run app_ui.py
API_BASE = os.getenv("MEDIDATA_API", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE}/extract-from-file"
HEALTH_URL = f"{API_BASE}/health"

# Large scanned bills take a while: render + OCR + several API calls.
REQUEST_TIMEOUT_S = int(os.getenv("MEDIDATA_TIMEOUT", "300"))

st.set_page_config(page_title="MediData", layout="wide")
st.title("MediData: A Medical Invoice Analyser")
st.markdown("Upload a medical bill PDF/image and extract structured data using AI.")

# ── Backend status ─────────────────────────────────────────────────────────
# Checked up front so a misconfigured backend is obvious before the user
# uploads a file and waits, rather than surfacing as a timeout afterwards.
with st.sidebar:
    st.subheader("Backend")
    st.caption(API_BASE)
    try:
        health = requests.get(HEALTH_URL, timeout=5).json()
        st.success("connected")
        st.write(f"**model:** `{health.get('model')}`")
        if health.get("mock_mode"):
            st.warning("MOCK MODE is on — results are dummy data.")
    except requests.RequestException as e:
        st.error("cannot reach backend")
        st.caption(str(e)[:200])
        st.info("Start it with:\n\n`uvicorn app.main:app --port 8000`")

uploaded_file = st.file_uploader("Upload Bill (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.success("File uploaded successfully!")

    if st.button("Extract Data", type="primary"):
        with st.spinner("Rendering pages, running OCR, calling the model…"):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(),
                                    uploaded_file.type)},
                    timeout=REQUEST_TIMEOUT_S,
                )
            except requests.Timeout:
                st.error(f"Timed out after {REQUEST_TIMEOUT_S}s. Large scanned "
                         f"bills can exceed this — raise MEDIDATA_TIMEOUT.")
                st.stop()
            except requests.RequestException as e:
                st.error(f"Could not reach {API_BASE}")
                st.caption(str(e)[:300])
                st.stop()

        if response.status_code != 200:
            st.error(f"API returned {response.status_code}")
            st.code(response.text[:1000])
            st.stop()

        result = response.json()

        if not result.get("is_success"):
            st.error("Extraction failed")
            st.code(result.get("error") or "no error detail returned")
            st.stop()

        data = result.get("data") or {}

        # Partial failure: some pages returned, some did not.
        if result.get("error"):
            st.warning(result["error"])

        st.subheader("Extraction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Items", data.get("total_item_count", 0))
        c2.metric("Extracted Total", f"₹ {data.get('grand_total', 0):,.2f}")

        # ── Reconciliation ─────────────────────────────────────────────────
        # The headline signal: the bill states its own total, so a mismatch
        # proves the extraction is wrong without needing any labelled data.
        rec = data.get("reconciliation") or {}
        matches = rec.get("matches")
        if matches is True:
            c3.metric("Printed Total", f"₹ {rec.get('printed_total', 0):,.2f}", "verified")
            st.success(f"**Verified** — extracted items match the total printed "
                       f"on the bill (₹ {rec.get('printed_total', 0):,.2f}).")
        elif matches is False:
            diff = rec.get("difference") or 0
            c3.metric("Printed Total", f"₹ {rec.get('printed_total', 0):,.2f}",
                      f"{diff:+,.2f}", delta_color="inverse")
            st.error(f"**Mismatch ({rec.get('pct_difference')}%)** — {rec.get('note')}")
        else:
            c3.metric("Printed Total", "—")
            st.info(rec.get("note") or "No printed total to verify against.")

        flags = data.get("fraud_flags") or []
        if flags:
            with st.expander(f"{len(flags)} fraud flag(s)", expanded=True):
                for f in flags:
                    st.warning(f)

        for page in data.get("pagewise_line_items", []):
            items = page.get("bill_items") or []
            st.markdown(f"### Page {page['page_no']} — {page.get('page_type', '')}"
                        f"  ·  {len(items)} item(s)")
            if items:
                df = pd.DataFrame(items)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No items on this page "
                        "(summary pages are suppressed to avoid double-counting).")

        usage = result.get("token_usage") or {}
        st.caption(f"tokens — in {usage.get('input_tokens', 0):,} / "
                   f"out {usage.get('output_tokens', 0):,} / "
                   f"total {usage.get('total_tokens', 0):,}")

        with st.expander("Raw JSON"):
            st.json(result)
