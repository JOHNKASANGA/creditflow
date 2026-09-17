import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import streamlit as st
import matplotlib.pyplot as plt
try:
    from face_verify import verify_face
    FACE_VERIFICATION_AVAILABLE = True
except Exception:
    FACE_VERIFICATION_AVAILABLE = False
import streamlit as st

# Make Cloud secrets visible to modules that read os.environ at import time
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass
from ocr.receipt_parser import parse_receipt, load_and_correct
from model.predict import score as score_features
from schema import FEATURE_RANGES
from face_verify import verify_face
from db import (
    init_db, create_user, verify_user, create_submission, get_pending_submissions,
    get_submission, save_score, get_all_scored_submissions, get_my_submissions,
    request_loan, get_loan_requests, approve_loan, reject_loan,
    create_session, get_session, delete_session,
    get_profile, save_profile, get_officer_visible_profile, update_password,
    set_phone_verified
)
from statistics import median
import random
import re

AMOUNT_THRESHOLD = 1000.0   # below this, a number is a quantity, not a Naira amount

def normalise_ng_phone(raw):
    """Accepts 08012345678, 8012345678, +2348012345678. Returns +234... or None."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("234") and len(digits) == 13:
        return "+" + digits
    if digits.startswith("0") and len(digits) == 11:
        return "+234" + digits[1:]
    if len(digits) == 10:
        return "+234" + digits
    return None


def new_otp():
    return f"{random.randint(0, 999999):06d}"

def verify_face(known_image_path, live_image_bytes):
        return False, "Face verification is unavailable in this deployment."

def suggest_from_numbers(numbers_found):
    """
    Split OCR numbers into likely quantities vs likely amounts, then suggest
    a daily revenue (median of the amounts — robust to one large restock line)
    and a restock amount (the largest amount, if it stands clearly apart).
    Suggestions only; the officer confirms both.
    """
    amounts = sorted(float(n) for n in numbers_found if n >= AMOUNT_THRESHOLD)
    if not amounts:
        return 0.0, 50000.0, []
    rev = float(median(amounts))
    top = amounts[-1]
    restock = top if top > rev * 1.5 else 50000.0
    return rev, restock, amounts

@st.cache_resource
def _warm_face_model():
    if not FACE_VERIFICATION_AVAILABLE:
        return False
    from face_verify import warm_up
    return warm_up()

st.set_page_config(page_title="CreditFlow AI", layout="wide")
init_db()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_DIR = os.path.join(ROOT_DIR, "profile_photos")
os.makedirs(PHOTO_DIR, exist_ok=True)

NAVY = "#0B1F3A"
ORANGE = "#F15A24"

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
.cf-bubble {
  display:flex; align-items:center; gap:10px;
  padding:8px 12px; border-radius:999px;
  background:#FFFFFF; border:1px solid #E4E9F0;
  transition: background .15s ease, box-shadow .15s ease, transform .15s ease;
  cursor:pointer; margin-bottom:6px;
}
.cf-bubble:hover {
  background:#FFF3EE; border-color:#F15A24;
  box-shadow:0 4px 12px rgba(241,90,36,0.18);
  transform: translateY(-1px);
}
.cf-avatar {
  width:38px; height:38px; border-radius:50%; flex:0 0 38px;
  background:#0B1F3A; color:#FFFFFF; font-weight:700; font-size:14px;
  display:flex; align-items:center; justify-content:center;
  background-size:cover; background-position:center;
}
.cf-bubble-name { font-weight:600; color:#0B1F3A; font-size:14px; line-height:1.2; }
.cf-bubble-role { color:#8A94A6; font-size:11px; letter-spacing:.04em; }
section[data-testid="stSidebar"] .stButton > button {
  width:100%; text-align:left; justify-content:flex-start; font-weight:500;
  border-radius:10px;
}
.cf-rank-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 12px; border-radius:10px;
  border:1px solid transparent;
  transition: background .15s ease, border-color .15s ease, transform .15s ease;
}
.cf-rank-row:hover {
  background:#FFF3EE; border-color:#F15A24; transform:translateX(2px);
}
.cf-auth-head { text-align:center; margin-bottom:6px; }
.cf-auth-head h3 { margin:0; color:#0B1F3A; font-size:20px; font-weight:700; }
.cf-auth-head p { margin:4px 0 0; color:#8A94A6; font-size:13px; }
</style>
<div style="background:#0B1F3A; padding:22px 28px; border-radius:14px; margin-bottom:24px;">
  <div style="color:#F15A24; font-size:28px; font-weight:700;">CreditFlow AI</div>
  <div style="color:#E8ECF2; font-size:14px; margin-top:4px;">
    Alternative credit scoring for informal MSMEs — built for Access Bank
  </div>
</div>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None
    st.session_state.username = None

# ---- Restore session from URL token (survives browser refresh) ----
if not st.session_state.authenticated:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_register = st.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.container(border=True):
                st.markdown(
                    '<div class="cf-auth-head"><h3>Welcome back</h3>'
                    '<p>Sign in to your CreditFlow account</p></div>',
                    unsafe_allow_html=True)
                if st.session_state.get("auth_flash"):
                    st.success(st.session_state.pop("auth_flash"))
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.button("Sign in", use_container_width=True, type="primary"):
                    role = verify_user(username, password)
                    if role:
                        token = create_session(username, role)
                        st.query_params["session"] = token
                        st.session_state.authenticated = True
                        st.session_state.role = role
                        st.session_state.username = username
                        st.session_state.page = "home"
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
        with tab_register:
            if "reg_stage" not in st.session_state:
                st.session_state.reg_stage = "details"

            if st.session_state.reg_stage == "details":
                with st.container(border=True):
                    st.markdown(
                        '<div class="cf-auth-head"><h3>Create your account</h3>'
                        '<p>For business owners applying for credit</p></div>',
                        unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        full_name = st.text_input("Full name")
                        new_username = st.text_input("Username")
                        new_password = st.text_input("Password", type="password")
                    with c2:
                        phone_raw = st.text_input("Phone number", placeholder="08012345678")
                        email = st.text_input("Email")
                        confirm_password = st.text_input("Confirm password", type="password")

                    venture_name = st.text_input("Business name")

                    st.caption("We'll send a 6-digit code to your phone to confirm it's yours.")

                    if st.button("Continue", use_container_width=True, type="primary"):
                        phone = normalise_ng_phone(phone_raw)
                        if not all([full_name, new_username, new_password, phone_raw, venture_name]):
                            st.error("Full name, username, password, phone number and business name are required.")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")
                        elif len(new_password) < 6:
                            st.error("Use a password of at least 6 characters.")
                        elif phone is None:
                            st.error("Enter a valid Nigerian phone number, e.g. 08012345678.")
                        else:
                            st.session_state.reg_data = {
                                "full_name": full_name, "username": new_username,
                                "password": new_password, "phone": phone,
                                "email": email, "venture_name": venture_name,
                            }
                            st.session_state.reg_otp = new_otp()
                            st.session_state.reg_stage = "verify"
                            st.rerun()

            else:
                data = st.session_state.reg_data
                with st.container(border=True):
                    st.markdown(
                        '<div class="cf-auth-head"><h3>Verify your phone</h3>'
                        f'<p>Enter the 6-digit code sent to {data["phone"]}</p></div>',
                        unsafe_allow_html=True)

                    st.info(f"Demo mode — your code is **{st.session_state.reg_otp}**. "
                            "A production build delivers this by SMS.")

                    code = st.text_input("6-digit code", max_chars=6)

                    v1, v2 = st.columns(2)
                    if v1.button("Verify and create account", use_container_width=True, type="primary"):
                        if code.strip() != st.session_state.reg_otp:
                            st.error("That code is incorrect.")
                        elif not create_user(data["username"], data["password"], "msme"):
                            st.error("That username is already taken.")
                            st.session_state.reg_stage = "details"
                        else:
                            save_profile(data["username"], {
                                "full_name": data["full_name"],
                                "venture_name": data["venture_name"],
                                "nin": "", "account_number": "", "gender": "",
                                "date_of_birth": "", "email": data["email"],
                                "phone_number": data["phone"], "address": "",
                            })
                            set_phone_verified(data["username"])
                            st.session_state.auth_flash = "Account created — please sign in."
                            st.session_state.reg_stage = "details"
                            st.session_state.pop("reg_data", None)
                            st.session_state.pop("reg_otp", None)
                            st.rerun()
                    if v2.button("Back", use_container_width=True):
                        st.session_state.reg_stage = "details"
                         st.rerun()
    st.stop()

# ---------------- Login / Register ----------------
if not st.session_state.authenticated:
    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    nav_col1, nav_col2 = st.columns(2)
    if nav_col1.button("Log in", use_container_width=True):
        st.session_state.auth_view = "login"
        st.rerun()
    if nav_col2.button("Register", use_container_width=True):
        st.session_state.auth_view = "register"
        st.rerun()

    if st.session_state.auth_view == "login":
        with st.container(border=True):
            if st.session_state.get("auth_flash"):
                st.success(st.session_state.pop("auth_flash"))
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign in", use_container_width=True):
                role = verify_user(username, password)
                if role:
                    token = create_session(username, role)
                    st.query_params["session"] = token
                    st.session_state.authenticated = True
                    st.session_state.role = role
                    st.session_state.username = username
                    st.session_state.page = "home"
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
    else:
        with st.container(border=True):
            new_username = st.text_input("Choose a username", key="reg_user")
            new_password = st.text_input("Choose a password", type="password", key="reg_pass")
            role_choice = st.selectbox("Account type", ["MSME", "Access Bank Officer"])
            if st.button("Create account", use_container_width=True):
                role_val = "msme" if role_choice == "MSME" else "officer"
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                elif create_user(new_username, new_password, role_val):
                    st.session_state.auth_view = "login"
                    st.session_state.auth_flash = "Account created — please log in."
                    st.rerun()
                else:
                    st.error("That username is already taken.")

    st.stop()

# ================= Shared helpers =================
STATUS_LABELS = {
    "pending": "Awaiting review",
    "scored": "Reviewed — loan available",
    "requested": "Loan requested — awaiting officer decision",
    "approved": "Approved — sent to bank for disbursement",
    "rejected": "Loan request rejected",
}
STATUS_COLORS = {
    "pending": "#8A94A6", "scored": "#1F9D55", "requested": "#F0A202",
    "approved": "#1F9D55", "rejected": "#D64545",
}
BADGE_COLORS = {"Low": "#1F9D55", "Medium": "#F0A202", "High": "#D64545"}
PROFILE_FIELDS = ["full_name", "venture_name", "nin", "account_number", "gender",
                  "date_of_birth", "email", "phone_number", "address"]

if "page" not in st.session_state:
    st.session_state.page = "home"


def img_data_uri(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return None


def initials(name, fallback):
    source = (name or fallback or "?").strip()
    parts = [p for p in source.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def render_score_circle(score, caption="Credit score"):
    if score is None:
        inner = '<div style="font-size:13px;color:#8A94A6;text-align:center;padding:0 12px;">Not scored yet</div>'
    else:
        inner = (f'<div style="font-size:40px;font-weight:700;color:{NAVY};line-height:1;">{score}</div>'
                 f'<div style="font-size:11px;color:#8A94A6;margin-top:4px;">/ 850</div>')
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px;">
      <div style="width:168px;height:168px;border-radius:50%;background:#FFFFFF;
        border:6px solid {ORANGE};display:flex;flex-direction:column;
        align-items:center;justify-content:center;
        box-shadow:0 4px 16px rgba(11,31,58,0.10);">
        {inner}
      </div>
      <div style="color:#8A94A6;font-size:12px;letter-spacing:.05em;text-transform:uppercase;">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def latest_scored_row(username):
    for row in get_my_submissions(username):
        if row["score"] is not None:
            return row
    return None


def profile_to_data(profile):
    return {f: ((profile or {}).get(f) or "") for f in PROFILE_FIELDS}


def save_photo(username, suffix, data_bytes):
    path = os.path.join(PHOTO_DIR, f"{username}_{suffix}.jpg")
    with open(path, "wb") as f:
        f.write(data_bytes)
    return path


def status_pill(status):
    c = STATUS_COLORS.get(status, "#8A94A6")
    return (f'<span style="background:{c}20;color:{c};padding:3px 12px;border-radius:999px;'
            f'font-size:12px;font-weight:600;border:1px solid {c};">'
            f'{STATUS_LABELS.get(status, status)}</span>')


def applicant_card(username):
    """Restricted view — only fields an officer is permitted to see."""
    p = get_officer_visible_profile(username)
    with st.container(border=True):
        if not p:
            st.caption(f"{username} — no profile submitted.")
            return
        c_img, c_info = st.columns([1, 3])
        uri = img_data_uri(p.get("photo_path"))
        with c_img:
            if uri:
                st.markdown(
                    f'<div style="width:82px;height:82px;border-radius:12px;background-image:url({uri});'
                    f'background-size:cover;background-position:center;"></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="width:82px;height:82px;border-radius:12px;background:{NAVY};color:#fff;'
                    f'display:flex;align-items:center;justify-content:center;font-weight:700;">'
                    f'{initials(p.get("full_name"), username)}</div>',
                    unsafe_allow_html=True)
        with c_info:
            st.markdown(f"**{p.get('full_name') or username}**")
            st.caption(p.get("venture_name") or "—")
            st.caption(f"Account: {p.get('account_number') or '—'}  ·  Gender: {p.get('gender') or '—'}")


# ================= Auto-refreshing fragments =================
@st.fragment(run_every=5)
def frag_my_applications():
    my_subs = get_my_submissions(st.session_state.username)
    if not my_subs:
        st.info("No applications yet. Upload a receipt to get started.")
        return
    for row in my_subs:
        with st.container(border=True):
            cols = st.columns([2, 1, 1, 2])
            cols[0].markdown(f"**Submission #{row['id']}**  \n{row['submitted_at'][:16]}")
            if row["score"] is not None:
                cols[1].metric("Score", row["score"])
                cols[2].metric("Max loan", f"NGN {row['max_loan']:,}")
            else:
                cols[1].caption("—")
                cols[2].caption("—")
            cols[3].markdown(status_pill(row["status"]), unsafe_allow_html=True)
            if row["requested_amount"]:
                cols[3].caption(f"Requested: NGN {row['requested_amount']:,.0f}")


@st.fragment(run_every=5)
def frag_leaderboard():
    st.subheader("All applicants — ranked")
    scored = get_all_scored_submissions()
    with st.container(border=True):
        if not scored:
            st.caption("No scored applicants yet.")
            return
        for rank, row in enumerate(scored, start=1):
            c = BADGE_COLORS.get(row["risk"], "#8A94A6")
            r_row, r_btn = st.columns([5, 1])
            with r_row:
                st.markdown(f"""
                <div class="cf-rank-row">
                  <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:22px; color:#8A94A6; font-weight:700; font-size:13px;">{rank}</div>
                    <div style="font-weight:600; color:#0B1F3A; font-size:14px;">{row['msme_username']}</div>
                  </div>
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-weight:700; color:#0B1F3A;">{row['score']}</span>
                    <span style="background:{c}20; color:{c}; padding:2px 10px; border-radius:999px;
                        font-size:11px; font-weight:600; border:1px solid {c};">{row['risk']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with r_btn:
                if st.button("View", key=f"lb_{row['id']}", use_container_width=True):
                    current = st.session_state.get("lb_open")
                    st.session_state.lb_open = None if current == row["id"] else row["id"]
                    st.rerun()

            if st.session_state.get("lb_open") == row["id"]:
                applicant_card(row["msme_username"])
                d1, d2, d3 = st.columns(3)
                d1.metric("Score", row["score"])
                d2.metric("Max loan", f"NGN {row['max_loan']:,}")
                d3.markdown(status_pill(row["status"]), unsafe_allow_html=True)

@st.fragment(run_every=5)
def frag_loan_requests():
    requests = get_loan_requests()
    if not requests:
        st.info("No pending loan requests.")
        return
    for row in requests:
        with st.container(border=True):
            applicant_card(row["msme_username"])
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("Score", row["score"])
            c2.metric("Requested", f"NGN {row['requested_amount']:,.0f}")
            with c3:
                a_col, r_col = st.columns(2)
                if a_col.button("Accept", key=f"accept_{row['id']}", use_container_width=True):
                    approve_loan(row["id"], st.session_state.username)
                    st.success("Sent to Access Bank for disbursement.")
                    st.rerun()
                if r_col.button("Reject", key=f"reject_{row['id']}", use_container_width=True):
                    reject_loan(row["id"], st.session_state.username)
                    st.warning("Loan request rejected.")
                    st.rerun()


# ================= MSME pages =================
def page_msme_home():
    st.subheader("Overview")
    latest = latest_scored_row(st.session_state.username)
    c_score, c_status = st.columns([1, 2])
    with c_score:
        with st.container(border=True):
            render_score_circle(latest["score"] if latest else None)
            if latest:
                c = BADGE_COLORS.get(latest["risk"], "#8A94A6")
                st.markdown(
                    f'<div style="text-align:center;margin-top:10px;">'
                    f'<span style="background:{c}20;color:{c};padding:5px 14px;border-radius:999px;'
                    f'font-weight:600;border:1px solid {c};">{latest["risk"]} risk</span></div>',
                    unsafe_allow_html=True)
                st.metric("Max recommended loan", f"NGN {latest['max_loan']:,}")
    with c_status:
        st.markdown("**Your loan activity**")
        frag_my_applications()


def page_msme_upload():
    st.subheader("Upload a receipt or logbook page")
    with st.container(border=True):
        uploaded = st.file_uploader("Image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded is not None:
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("Reading document..."):
                result = parse_receipt(temp_path)

            if result.get("error"):
                st.error(result["error"])
                return

            col_img, col_data = st.columns([1, 1])
            with col_img:
                st.image(load_and_correct(temp_path), use_container_width=True)
            with col_data:
                st.markdown("**Extracted text**")
                st.text_area("", result["raw_text"], height=140, label_visibility="collapsed")
                if result.get("date"):
                    st.caption(f"Date detected: {result['date']}")
                if st.button("Submit for review", use_container_width=True):
                    create_submission(st.session_state.username, result["raw_text"], result["numbers_found"])
                    st.success("Submitted. An Access Bank officer will review it.")


def page_msme_loan():
    st.subheader("Loan assessment")
    scored = [r for r in get_my_submissions(st.session_state.username) if r["score"] is not None]
    if not scored:
        st.info("No assessment yet. Once an officer reviews a submission, the AI grading appears here.")
        return

    options = {f"#{r['id']} — scored {r['score']} ({r['status']})": r["id"] for r in scored}
    label = st.selectbox("Select an assessment", list(options.keys()))
    sub = get_submission(options[label])

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            render_score_circle(sub["score"])
        with c2:
            c = BADGE_COLORS.get(sub["risk"], "#8A94A6")
            st.markdown(
                f'<div style="margin-top:10px;"><span style="background:{c}20;color:{c};padding:6px 16px;'
                f'border-radius:999px;font-weight:600;border:1px solid {c};">{sub["risk"]} risk</span></div><br>',
                unsafe_allow_html=True)
            st.metric("Max recommended loan", f"NGN {sub['max_loan']:,}")
            st.markdown(status_pill(sub["status"]), unsafe_allow_html=True)
        with c3:
            st.caption("Assessed on")
            st.write((sub["scored_at"] or "")[:16] or "—")

    features = {
        "daily_revenue": sub["daily_revenue"], "restock_frequency": sub["restock_frequency"],
        "restock_amount": sub["restock_amount"], "pos_sales_consistency": sub["pos_sales_consistency"],
        "supplier_payment_delay": sub["supplier_payment_delay"],
        "revenue_volatility": sub["revenue_volatility"], "months_operating": sub["months_operating"],
    }
    if all(v is not None for v in features.values()):
        with st.container(border=True):
            st.markdown("**How the AI graded you**")
            st.caption("Green bars pushed your score up; red bars pulled it down.")
            detail = score_features(features)
            items = sorted(detail["explanation"].items(), key=lambda x: x[1])
            labels = [k.replace("_", " ").title() for k, _ in items]
            values = [v for _, v in items]
            colors = ["#1F9D55" if v >= 0 else "#D64545" for v in values]
            fig, ax = plt.subplots(figsize=(7, 3.0))
            ax.barh(labels, values, color=colors)
            ax.axvline(0, color=NAVY, linewidth=0.8)
            ax.set_xlabel("Impact on credit score", fontsize=9)
            ax.tick_params(labelsize=9)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

    if sub["status"] != "scored":
        st.info("A loan request has already been made against this assessment.")
        return

        st.markdown("### Request a loan")
    profile = get_profile(st.session_state.username)
    with st.container(border=True):
        amount = st.number_input(
            "Amount requested (NGN)", min_value=0.0,
            max_value=float(sub["max_loan"]), value=float(sub["max_loan"])
        )

        if FACE_VERIFICATION_AVAILABLE:
            st.markdown("**Identity verification**")
            st.caption("Take a live selfie to confirm this request is coming from you.")
            live_capture = st.camera_input("Verify your face", key="loan_face_verify")
        else:
            live_capture = None
            st.caption("Identity verification runs in the full deployment.")

        if st.button("Submit loan request", use_container_width=True):
            if not FACE_VERIFICATION_AVAILABLE:
                request_loan(sub["id"], amount)
                st.success("Loan request submitted for review.")
                st.rerun()
            elif live_capture is None:
                st.error("Take a selfie to verify your identity before submitting.")
            elif not profile or not profile.get("biometric_path"):
                st.error("No registered biometric photo on file. Add one under Settings first.")
            else:
                with st.spinner("Verifying identity..."):
                    verified, message = verify_face(profile["biometric_path"], live_capture.getvalue())
                if verified:
                    request_loan(sub["id"], amount)
                    st.success("Identity verified. Loan request submitted for review.")
                    st.rerun()
                else:
                    st.error(message)

def page_profile():
    profile = get_profile(st.session_state.username) if st.session_state.role == "msme" else None
    display_name = (profile or {}).get("full_name") or st.session_state.username
    face_uri = img_data_uri((profile or {}).get("biometric_path")) or img_data_uri((profile or {}).get("photo_path"))

    c_left, c_right = st.columns([1, 1])
    with c_left:
        with st.container(border=True):
            if face_uri:
                st.markdown(
                    f'<div style="display:flex;justify-content:center;">'
                    f'<div style="width:168px;height:168px;border-radius:50%;background-image:url({face_uri});'
                    f'background-size:cover;background-position:center;border:6px solid {NAVY};"></div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display:flex;justify-content:center;">'
                    f'<div style="width:168px;height:168px;border-radius:50%;background:{NAVY};color:#fff;'
                    f'font-size:44px;font-weight:700;display:flex;align-items:center;justify-content:center;">'
                    f'{initials(display_name, st.session_state.username)}</div></div>',
                    unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:center;margin-top:14px;">'
                f'<div style="font-size:22px;font-weight:700;color:{NAVY};">{display_name}</div>'
                f'<div style="color:#8A94A6;font-size:12px;letter-spacing:.06em;text-transform:uppercase;">'
                f'{st.session_state.role}</div></div>',
                unsafe_allow_html=True)
            if profile and profile.get("venture_name"):
                st.markdown(
                    f'<div style="text-align:center;color:#5A6577;margin-top:6px;">{profile["venture_name"]}</div>',
                    unsafe_allow_html=True)

    with c_right:
        with st.container(border=True):
            if st.session_state.role == "msme":
                latest = latest_scored_row(st.session_state.username)
                render_score_circle(latest["score"] if latest else None)
            else:
                st.caption("Officer accounts are not credit scored.")

    if st.session_state.role != "msme":
        return

    st.markdown("### Business details")
    data = profile_to_data(profile)
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full name", value=data["full_name"])
            venture_name = st.text_input("Business / venture name", value=data["venture_name"])
            gender = st.text_input("Gender", value=data["gender"])
            date_of_birth = st.text_input("Date of birth", value=data["date_of_birth"])
            email = st.text_input("Email", value=data["email"])
        with c2:
            nin = st.text_input("NIN", value=data["nin"])
            account_number = st.text_input("Account number", value=data["account_number"])
            phone_number = st.text_input("Phone number", value=data["phone_number"])
            address = st.text_area("Address", value=data["address"], height=96)
        if st.form_submit_button("Save profile", use_container_width=True):
            save_profile(st.session_state.username, {
                "full_name": full_name, "venture_name": venture_name, "nin": nin,
                "account_number": account_number, "gender": gender,
                "date_of_birth": date_of_birth, "email": email,
                "phone_number": phone_number, "address": address,
            })
            st.success("Profile saved.")
            st.rerun()


def page_settings():
    st.subheader("Settings")

    with st.container(border=True):
        st.markdown("**Change password**")
        current_pw = st.text_input("Current password", type="password", key="set_cur_pw")
        new_pw = st.text_input("New password", type="password", key="set_new_pw")
        confirm_pw = st.text_input("Confirm new password", type="password", key="set_conf_pw")
        if st.button("Update password"):
            if new_pw != confirm_pw:
                st.error("New passwords do not match.")
            else:
                ok, msg = update_password(st.session_state.username, current_pw, new_pw)
                st.success(msg) if ok else st.error(msg)

    if st.session_state.role != "msme":
        return

    profile = get_profile(st.session_state.username)

    with st.container(border=True):
        st.markdown("**National Identification Number**")
        if not profile or not profile.get("nin"):
            st.caption("No NIN on file. Add one under Profile.")
        else:
            if st.checkbox("Reveal my NIN", key="reveal_nin"):
                st.code(profile["nin"])
            else:
                st.code("•" * len(profile["nin"]))
            st.caption("Stored encrypted at rest.")

    with st.container(border=True):
        st.markdown("**Profile picture**")
        new_photo = st.file_uploader("Upload a profile photo", type=["jpg", "jpeg", "png"], key="photo_up")
        if new_photo is not None and st.button("Save profile picture"):
            path = save_photo(st.session_state.username, "photo", new_photo.getbuffer())
            save_profile(st.session_state.username, profile_to_data(profile), photo_path=path)
            st.success("Profile picture updated.")
            st.rerun()

    with st.container(border=True):
        st.markdown("**Biometric face registration**")
        st.caption("This is the reference photo your live selfie is matched against when you request a loan.")
        if profile and profile.get("biometric_path"):
            st.caption("A biometric photo is currently registered. Capturing a new one replaces it.")
        biometric = st.camera_input("Capture your face", key="biometric_capture")
        if biometric is not None and st.button("Register this face"):
            path = save_photo(st.session_state.username, "biometric", biometric.getvalue())
            save_profile(st.session_state.username, profile_to_data(profile), biometric_path=path)
            st.success("Biometric photo registered.")
            st.rerun()


def page_help():
    st.subheader("Help & Support")
    with st.container(border=True):
        st.markdown("**How CreditFlow works**")
        st.write(
            "Upload a photo of a receipt or logbook page. CreditFlow reads it automatically, "
            "an Access Bank officer reviews the figures, and the AI produces a credit score "
            "between 300 and 850 along with a recommended loan limit. Once scored, you can "
            "request a loan — confirming your identity with a live selfie — and an officer "
            "approves or declines it."
        )
    with st.container(border=True):
        st.markdown("**Common questions**")
        with st.expander("Why was my receipt not read correctly?"):
            st.write("Photograph the page flat, in good light, with the whole page in frame. "
                     "Blurred or angled photos reduce accuracy.")
        with st.expander("Why does my loan request need a selfie?"):
            st.write("It confirms the request is coming from the registered account holder. "
                     "Register your reference photo under Settings → Biometric face registration.")
        with st.expander("Who can see my personal details?"):
            st.write("Officers see only your name, business name, photo, account number and gender. "
                     "Your NIN, date of birth, email, phone and address are encrypted and not shown to them.")
    with st.container(border=True):
        st.markdown("**Contact**")
        st.write("Reach your Access Bank relationship officer, or email support@creditflow.example")


# ================= Officer pages =================
def page_officer_home():
    st.subheader("Overview")
    pending = get_pending_submissions()
    requests = get_loan_requests()
    scored = get_all_scored_submissions()
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Awaiting review", len(pending))
    with c2:
        with st.container(border=True):
            st.metric("Loan requests pending", len(requests))
    with c3:
        with st.container(border=True):
            st.metric("Applicants scored", len(scored))
    frag_leaderboard()

def page_officer_review():
    col_main, col_leaderboard = st.columns([2, 1])

    with col_main:
        st.subheader("Pending submissions")
        pending = get_pending_submissions()
        if not pending:
            st.info("No pending submissions.")
        else:
            options = {f'#{row["id"]} — {row["msme_username"]} ({row["submitted_at"][:16]})': row["id"]
                       for row in pending}
            selected_label = st.selectbox("Select a submission to review", list(options.keys()))
            sub_id = options[selected_label]
            sub = get_submission(sub_id)

            applicant_card(sub["msme_username"])

            with st.container(border=True):
                st.markdown("**Extracted text**")
                st.text_area("", sub["raw_text"] or "", height=100, label_visibility="collapsed")

            numbers_found = json.loads(sub["numbers_found"]) if sub["numbers_found"] else []
            sug_revenue, sug_restock, amounts = suggest_from_numbers(numbers_found)

            with st.container(border=True):
                st.markdown("**Read from the document**")
                st.caption(
                    "Suggested by the document parser from the figures on the page. Confirm or correct."
                )
                if amounts:
                    st.caption("Amounts detected: " + ", ".join(f"NGN {a:,.0f}" for a in amounts))
                else:
                    st.caption("No amounts detected on this page — enter both figures manually.")
                d1, d2 = st.columns(2)
                with d1:
                    daily_revenue = st.number_input(
                        "Daily revenue (NGN)", value=float(sug_revenue), min_value=0.0)
                with d2:
                    restock_amount = st.number_input(
                        "Restock amount (NGN)", value=float(sug_restock), min_value=0.0)

            with st.container(border=True):
                st.markdown("**Officer assessment**")
                st.caption(
                    "Relationship history a single document cannot show. Enter from your own "
                    "knowledge of the business."
                )
                o1, o2 = st.columns(2)
                with o1:
                    restock_frequency = st.number_input(
                        "Restock frequency (days)", value=7.0, min_value=1.0)
                    supplier_payment_delay = st.number_input(
                        "Supplier payment delay (days)", value=10.0, min_value=0.0)
                    months_operating = st.number_input(
                        "Months operating", value=12.0, min_value=0.0)
                with o2:
                    pos_sales_consistency = st.slider("POS sales consistency", 0.0, 1.0, 0.5)
                    revenue_volatility = st.number_input(
                        "Revenue volatility", value=0.5, min_value=0.0)

            calculate = st.button("Calculate credit score", use_container_width=True)

            if calculate:
                features = {
                    "daily_revenue": daily_revenue, "restock_frequency": restock_frequency,
                    "restock_amount": restock_amount, "pos_sales_consistency": pos_sales_consistency,
                    "supplier_payment_delay": supplier_payment_delay,
                    "revenue_volatility": revenue_volatility,
                    "months_operating": months_operating,
                }
                result = score_features(features)
                save_score(sub_id, features, result["score"], result["risk"], result["max_loan"],
                           st.session_state.username)

                c = BADGE_COLORS[result["risk"]]
                with st.container(border=True):
                    col_gauge, col_info, col_chart = st.columns([1, 1, 2])
                    with col_gauge:
                        render_score_circle(result["score"])
                    with col_info:
                        st.markdown(
                            f'<div style="margin-top:6px;"><span style="background:{c}20; color:{c}; '
                            f'padding:6px 16px; border-radius:999px; font-weight:600; border:1px solid {c};">'
                            f'{result["risk"]} risk</span></div><br>',
                            unsafe_allow_html=True
                        )
                        st.metric("Max recommended loan", f"NGN {result['max_loan']:,}")
                    with col_chart:
                        st.markdown("**Score breakdown**")
                        items = sorted(result["explanation"].items(), key=lambda x: x[1])
                        labels = [k.replace("_", " ").title() for k, _ in items]
                        values = [v for _, v in items]
                        colors = ["#1F9D55" if v >= 0 else "#D64545" for v in values]
                        fig, ax = plt.subplots(figsize=(6, 2.8))
                        ax.barh(labels, values, color=colors)
                        ax.axvline(0, color=NAVY, linewidth=0.8)
                        ax.set_xlabel("Impact on credit score", fontsize=9)
                        ax.tick_params(labelsize=9)
                        for spine in ["top", "right"]:
                            ax.spines[spine].set_visible(False)
                        fig.tight_layout()
                        st.pyplot(fig)
                st.success("Score saved.")

    with col_leaderboard:
        frag_leaderboard()
def page_officer_requests():
    st.subheader("Pending loan requests")
    frag_loan_requests()


# ================= Sidebar: profile bubble + navigation =================
_profile = get_profile(st.session_state.username) if st.session_state.role == "msme" else None
_display_name = (_profile or {}).get("full_name") or st.session_state.username
_avatar_uri = img_data_uri((_profile or {}).get("biometric_path")) or img_data_uri((_profile or {}).get("photo_path"))

if _avatar_uri:
    _avatar_html = f'<div class="cf-avatar" style="background-image:url({_avatar_uri});"></div>'
else:
    _avatar_html = f'<div class="cf-avatar">{initials(_display_name, st.session_state.username)}</div>'

with st.sidebar:
    st.markdown(f"""
    <div class="cf-bubble">
      {_avatar_html}
      <div>
        <div class="cf-bubble-name">{_display_name}</div>
        <div class="cf-bubble-role">{st.session_state.role.upper()}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View profile", key="nav_profile", use_container_width=True,
                 type="primary" if st.session_state.page == "profile" else "secondary"):
        st.session_state.page = "profile"
        st.rerun()

    st.divider()

    if st.session_state.role == "msme":
        nav_items = [("home", "Home"), ("upload", "Upload"), ("loan", "Loan"),
                     ("settings", "Settings"), ("help", "Help & Support")]
    else:
        nav_items = [("home", "Home"), ("review", "Review Submissions"), ("requests", "Loan Requests"),
                     ("settings", "Settings"), ("help", "Help & Support")]

    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.divider()
    if st.button("Refresh data", key="nav_refresh", use_container_width=True):
        st.rerun()
    if st.button("Log out", key="nav_logout", use_container_width=True):
        token = st.query_params.get("session")
        if token:
            delete_session(token)
        st.query_params.clear()
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.page = "home"
        st.rerun()

# ================= Router =================
PAGE = st.session_state.page
if PAGE == "profile":
    page_profile()
elif PAGE == "settings":
    page_settings()
elif PAGE == "help":
    page_help()
elif st.session_state.role == "msme":
    if PAGE == "upload":
        page_msme_upload()
    elif PAGE == "loan":
        page_msme_loan()
    else:
        page_msme_home()
else:
    if PAGE == "review":
        page_officer_review()
    elif PAGE == "requests":
        page_officer_requests()
    else:
        page_officer_home()