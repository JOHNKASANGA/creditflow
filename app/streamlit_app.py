import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt
from ocr.receipt_parser import parse_receipt, load_and_correct
from model.predict import score
from schema import FEATURE_RANGES

st.set_page_config(page_title="CreditFlow AI", layout="wide")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
body { background: #F4F6F8; }
[data-testid="stFileUploaderDropzone"] { padding: 0.5rem; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-radius: 16px;
  background: #FFFFFF;
  box-shadow: 0 1px 3px rgba(11,31,58,0.08), 0 6px 16px rgba(11,31,58,0.05);
}
</style>
<div style="background:#0B1F3A; padding:22px 28px; border-radius:14px; margin-bottom:24px;">
  <div style="color:#F15A24; font-size:28px; font-weight:700;">CreditFlow AI</div>
  <div style="color:#E8ECF2; font-size:14px; margin-top:4px;">
    Alternative credit scoring for informal MSMEs — built for Access Bank
  </div>
</div>
""", unsafe_allow_html=True)

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None

tab1, tab2 = st.tabs(["Step 1 — MSME Upload", "Step 2 — Officer Review & Score"])

with tab1:
    st.subheader("Upload a receipt or logbook page")
    with st.container(border=True):
        uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded is not None:
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("Reading document..."):
                result = parse_receipt(temp_path)
            st.session_state.ocr_result = result

            col_img, col_data = st.columns([1, 1])
            with col_img:
                st.image(load_and_correct(temp_path), use_container_width=True)
            with col_data:
                st.markdown('<i class="ti ti-file-text"></i> **Extracted text**', unsafe_allow_html=True)
                st.text_area("", result["raw_text"], height=140, label_visibility="collapsed")
                if result["numbers_found"]:
                    largest = max(result["numbers_found"])
                    st.caption(f"Largest figure detected: NGN {largest:,.0f} — auto-filled as daily revenue in Step 2 (adjustable).")
                else:
                    st.caption("No numeric figures detected — enter values manually in Step 2.")
    st.info("Once uploaded, move to Step 2 to review and score.")

with tab2:
    st.subheader("Review figures and calculate score")
    ocr = st.session_state.ocr_result
    guessed_revenue = max(ocr["numbers_found"]) if ocr and ocr["numbers_found"] else FEATURE_RANGES["daily_revenue"][0]

    with st.container(border=True):
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

        calculate = st.button("Calculate credit score", use_container_width=True)

    if calculate:
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
        badge_colors = {"Low": "#1F9D55", "Medium": "#F0A202", "High": "#D64545"}
        c = badge_colors[result["risk"]]
        pct = min(max((result["score"] - 300) / 550 * 100, 0), 100)

        with st.container(border=True):
            col_gauge, col_info, col_chart = st.columns([1, 1, 2])

            with col_gauge:
                st.markdown(f"""
                <div style="width:150px;height:150px;border-radius:50%;
                  background:conic-gradient({c} {pct}%, #E9ECEF 0);
                  display:flex;align-items:center;justify-content:center;margin:auto;">
                  <div style="width:114px;height:114px;border-radius:50%;background:white;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
                    <div style="font-size:30px;font-weight:700;color:#0B1F3A;">{result['score']}</div>
                    <div style="font-size:11px;color:#8A94A6;">/ 850</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col_info:
                st.markdown(
                    f'<div style="margin-top:6px;"><i class="ti ti-shield-check"></i> '
                    f'<span style="background:{c}20; color:{c}; padding:6px 16px; '
                    f'border-radius:999px; font-weight:600; border:1px solid {c};">'
                    f'{result["risk"]} risk</span></div><br>', unsafe_allow_html=True
                )
                st.metric("Max recommended loan", f"NGN {result['max_loan']:,}")

            with col_chart:
                st.markdown('<i class="ti ti-chart-bar"></i> **Score breakdown**', unsafe_allow_html=True)
                items = sorted(result["explanation"].items(), key=lambda x: x[1])
                labels = [k.replace("_", " ").title() for k, _ in items]
                values = [v for _, v in items]
                colors = ["#1F9D55" if v >= 0 else "#D64545" for v in values]

                fig, ax = plt.subplots(figsize=(6, 2.8))
                ax.barh(labels, values, color=colors)
                ax.axvline(0, color="#0B1F3A", linewidth=0.8)
                ax.set_xlabel("Impact on credit score", fontsize=9)
                ax.tick_params(labelsize=9)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)