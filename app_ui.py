import os

import pandas as pd
import requests
import streamlit as st

# Where the FastAPI backend lives. Checked in priority order:
#
#   1. st.secrets      - Streamlit Cloud. Secrets are NOT exported as environment
#                        variables there, so os.getenv alone would miss them and
#                        the deployed app would try to reach its own container.
#   2. MEDIDATA_API    - normal env var, for local runs and Docker
#   3. localhost       - sensible default when running on your own machine
#
# On Streamlit Cloud: Manage app -> Settings -> Secrets, then add
#     MEDIDATA_API = "http://<your-backend-host>:8000"
def _secret(name: str) -> str | None:
    """Read from Streamlit secrets first, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # no secrets configured - normal when running locally
    return os.getenv(name)


API_BASE = (_secret("MEDIDATA_API") or "").rstrip("/")

# Two ways to run:
#
#   DIRECT MODE (no MEDIDATA_API set) - this app does the extraction itself, in
#   this same process. Nothing else to deploy and nothing else to keep alive.
#   This is what Streamlit Cloud uses.
#
#   API MODE (MEDIDATA_API set) - forward uploads to a separate FastAPI server.
#   Useful when the API runs elsewhere, or you want the UI to stay thin.
DIRECT_MODE = not API_BASE

_extractor = None
_import_error = None

if DIRECT_MODE:
    # The extractor reads its config from the environment at import time, so
    # secrets must be copied across BEFORE importing it.
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_MODEL_FALLBACKS",
                "USE_MOCK_MODE", "BATCH_SIZE", "PDF_DPI"):
        value = _secret(key)
        if value is not None:
            os.environ[key] = value

    try:
        from app.extractor import BillExtractor

        @st.cache_resource(show_spinner=False)
        def _get_extractor():
            """Built once per session - loading it per upload would be wasteful."""
            return BillExtractor()

        _extractor = _get_extractor()
    except Exception as e:  # noqa: BLE001
        _import_error = e
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

    if DIRECT_MODE:
        st.caption("running in this app (no separate server)")
        if _import_error:
            st.error("extractor failed to load")
            st.caption(str(_import_error)[:300])
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is not set")
            st.caption("Streamlit Cloud: Manage app → Settings → Secrets")
        else:
            st.success("ready")
            st.write(f"**model:** `{os.getenv('GEMINI_MODEL', 'gemini-flash-latest')}`")
            if os.getenv("USE_MOCK_MODE", "").lower() == "true":
                st.warning("MOCK MODE is on — results are dummy data.")

    else:
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

    # Shows WHERE config was found, without ever printing the key itself.
    # Without this, a missing secret looks identical to a broken import.
    with st.expander("config check"):
        try:
            secret_keys = sorted(st.secrets.keys())
            st.write(f"secrets found: `{secret_keys or 'none'}`")
        except Exception:
            st.write("secrets found: `none configured`")

        key = os.getenv("GOOGLE_API_KEY", "")
        st.write(f"GOOGLE_API_KEY: {'set, ' + str(len(key)) + ' chars' if key else '**MISSING**'}")
        st.write(f"GEMINI_MODEL: `{os.getenv('GEMINI_MODEL') or 'not set (using default)'}`")

        try:
            import google.generativeai  # noqa: F401
            st.write("google-generativeai: installed")
        except ImportError:
            st.write("google-generativeai: **NOT INSTALLED**")

        import shutil
        st.write(f"poppler (pdftoppm): {'found' if shutil.which('pdftoppm') else '**MISSING**'}")
        st.write(f"tesseract: {'found' if shutil.which('tesseract') else '**MISSING**'}")

uploaded_file = st.file_uploader("Upload Bill (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.success("File uploaded successfully!")

    if st.button("Extract Data", type="primary"):
        with st.spinner("Rendering pages, running OCR, calling the model…"):
            if DIRECT_MODE:
                if _extractor is None:
                    st.error("Extractor is not available — see the sidebar.")
                    st.stop()

                import tempfile
                from pathlib import Path

                suffix = Path(uploaded_file.name).suffix.lower() or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    result = _extractor.extract(tmp_path).model_dump()
                except Exception as e:  # noqa: BLE001
                    st.error("Extraction failed")
                    st.code(str(e)[:1000])
                    st.stop()
                finally:
                    os.unlink(tmp_path)

            else:
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
