import streamlit as st
import requests
import json

# 🔧 CONFIG
API_URL = "http://13.206.108.88:8000/extract-from-file"  # AWS URL
# API_URL = "http://127.0.0.1:8000/extract-from-file"   # Local

st.set_page_config(page_title="MediData", layout="wide")

st.title("MediData: A Medical Invoice Analyser")
st.markdown("Upload a medical bill PDF/image and extract structured data using AI.")

# 📂 File upload
uploaded_file = st.file_uploader(
    "Upload Bill (PDF/Image)",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    st.success("File uploaded successfully!")

    if st.button("🚀 Extract Data"):
        with st.spinner("Processing..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("📊 Extraction Result")

                    if result.get("is_success"):
                        data = result.get("data", {})

                        # 🔹 Show totals
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Total Items", data.get("total_item_count", 0))

                        with col2:
                            st.metric("Grand Total", f"₹ {data.get('grand_total', 0)}")

                        # 🔹 Page-wise items
                        for page in data.get("pagewise_line_items", []):
                            st.markdown(f"### 📄 Page {page['page_no']}")

                            items = page.get("bill_items", [])

                            if items:
                                st.dataframe(items, use_container_width=True)
                            else:
                                st.info("No items found")

                        # 🔹 Raw JSON
                        with st.expander("🔍 Raw JSON"):
                            st.json(result)

                    else:
                        st.error("Extraction failed")
                        st.json(result)

                else:
                    st.error(f"API Error: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")