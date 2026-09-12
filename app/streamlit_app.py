import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ocr.receipt_parser import parse_receipt
from model.predict import score
from schema import FEATURE_RANGES

st.title("CreditFlow AI")

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None

tab1, tab2 = st.tabs(["MSME Portal", "Access Bank Officer Portal"])

with tab1:
    st.header("Upload receipt / logbook page")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Reading receipt..."):
            result = parse_receipt(temp_path)
        st.session_state.ocr_result = result
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        st.text_area("Extracted text", result["raw_text"], height=100)
        st.write("Numbers found:", result["numbers_found"])

with tab2:
    st.header("Credit assessment")
    ocr = st.session_state.ocr_result
    guessed_revenue = max(ocr["numbers_found"]) if ocr and ocr["numbers_found"] else FEATURE_RANGES["daily_revenue"][0]
    st.caption("Fields are pre-filled from OCR where possible — review and adjust before scoring.")

    col1, col2 = st.columns(2)
    with col1:
        daily_revenue = st.number_input("Daily revenue (NGN)", value=float(guessed_revenue), min_value=0.0)
        restock_frequency = st.number_input("Restock frequency (days)", value=7.0, min_value=1.0)
        restock_amount = st.number_input("Restock amount (NGN)", value=50000.0, min_value=0.0)
        pos_sales_consistency = st.slider("POS sales consistency", 0.0, 1.0, 0.5)
    with col2:
        supplier_payment_delay = st.number_input("Supplier payment delay (days)", value=10.0, min_value=0.0)
        revenue_volatility = st.number_input("Revenue volatility", value=0.5, min_value=0.0)
        months_operating = st.number_input("Months operating", value=12.0, min_value=0.0)

    if st.button("Calculate credit score"):
        features = {
            "daily_revenue": daily_revenue,
            "restock_frequency": restock_frequency,
            "restock_amount": restock_amount,
            "pos_sales_consistency": pos_sales_consistency,
            "supplier_payment_delay": supplier_payment_delay,
            "revenue_volatility": revenue_volatility,
            "months_operating": months_operating,
        }
        result = score(features)
        st.metric("Credit score", result["score"])
        st.metric("Risk classification", result["risk"])
        st.metric("Max recommended loan", f"NGN {result['max_loan']:,}")
        st.subheader("Score breakdown (SHAP)")
        st.bar_chart(result["explanation"])