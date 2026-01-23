import os
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from camera_input_live import camera_input_live

# -------------------- Configuration & Custom Styles --------------------
API_URL = os.getenv("API_URL", "http://api:8000")
st.set_page_config(page_title="ASL AI Vision Lab", layout="wide", page_icon="🤟")


def apply_pro_design():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(145deg, #f8f9fc 0%, #e2e8f0 100%); }

        section[data-testid="stSidebar"] { background-color: #111827 !important; }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 { color: #ffffff !important; }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }

        .hero-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 20px;
            border-left: 8px solid #4f46e5;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 10px;
        }
        .hero-text {
            color: #111827;
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            line-height: 1;
        }

        .tips-box {
            background: rgba(79, 70, 229, 0.05);
            border: 1px dashed #4f46e5;
            padding: 10px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_pro_design()

# -------------------- Session State Init --------------------
if "active" not in st.session_state:
    st.session_state.active = False
if "last_good_result" not in st.session_state:
    st.session_state.last_good_result = None
if "live_nohands" not in st.session_state:
    st.session_state.live_nohands = False
if "show_correction" not in st.session_state:
    st.session_state.show_correction = False
if "feedback_sent_for" not in st.session_state:
    st.session_state.feedback_sent_for = set()

# -------------------- Helpers --------------------
def health_check() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def img_to_base64_jpeg(img: Image.Image) -> str:
    img = img.convert("RGB")
    if img.width > 640:
        img = img.resize((640, int(img.height * (640 / img.width))))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_predict(image_b64: str, language: str = "en"):
    payload = {"image": image_b64, "language": language}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def send_feedback(job_id: str, accepted: bool, corrected_gesture: str = ""):
    payload = {"job_id": str(job_id), "accepted": accepted, "corrected_gesture": corrected_gesture}
    r = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text}")
    return r.json()


NO_HANDS_TOKENS = {"NO_HANDS", "NO HANDS", "NO_HAND", "NO HAND", "NONE", "UNKNOWN"}

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🤟 ASL Vision")
    page = st.radio("Navigation", ["Predict Lab", "Monitoring"], index=0)

    st.divider()
    st.subheader("⚙️ Settings")
    lang_label = st.selectbox("Output Language", ["English", "Dutch"])
    language = "en" if lang_label == "English" else "nl"

    st.divider()
    st.subheader("📡 Connection")
    ok = health_check()
    st.markdown(f"**API URL:** `{API_URL}`")
    status_icon = "🟢" if ok else "🔴"
    st.markdown(f"**Status:** {status_icon} {'Online' if ok else 'Offline'}")

# -------------------- Predict Page --------------------
if page == "Predict Lab":
    col_in, col_out = st.columns([1.2, 1], gap="large")

    with col_in:
        st.markdown("### 🎥 Capture Your Sign")
        st.caption("Show a sign to the camera and we’ll translate it in real time.")
        mode = st.tabs(["🎥 Live Stream", "📸 Webcam Photo", "📁 Upload Image"])

        # ---------------- Live Stream ----------------
        with mode[0]:
            c1, c2 = st.columns(2)
            if c1.button("▶ Start Prediction", use_container_width=True):
                st.session_state.active = True
            if c2.button("⏹ Stop & Disconnect", use_container_width=True):
                st.session_state.active = False
                st.session_state.last_good_result = None
                st.session_state.live_nohands = False
                st.session_state.show_correction = False

            if st.session_state.get("active"):
                live_img = camera_input_live()
                if live_img:
                    img = Image.open(live_img)
                    st.image(img, use_container_width=True)

                    b64 = img_to_base64_jpeg(img)
                    try:
                        result = call_predict(b64, language=language)
                        gesture_up = (result.get("gesture") or "").strip().upper()
                        st.session_state.live_nohands = gesture_up in NO_HANDS_TOKENS

                        if gesture_up not in NO_HANDS_TOKENS:
                            if float(result.get("confidence", 0) or 0) >= 0.5:
                                st.session_state.last_good_result = result
                                # ✅ Reset correction UI when a new prediction arrives
                                st.session_state.show_correction = False
                    except Exception as e:
                        st.caption(f"Live predict error: {e}")
            else:
                st.info("Live mode is inactive. Click Start to begin.")

        # ---------------- Webcam Snapshot ----------------
        with mode[1]:
            cam_shot = st.camera_input("Take a Snapshot")
            if cam_shot:
                img = Image.open(cam_shot)
                with st.spinner("Analyzing..."):
                    res = call_predict(img_to_base64_jpeg(img), language=language)
                    st.session_state.last_good_result = res
                    st.session_state.live_nohands = False
                    st.session_state.show_correction = False

        # ---------------- Upload ----------------
        with mode[2]:
            uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, width=300)
                if st.button("Predict Uploaded Image", type="primary"):
                    res = call_predict(img_to_base64_jpeg(img), language=language)
                    st.session_state.last_good_result = res
                    st.session_state.live_nohands = False
                    st.session_state.show_correction = False

        st.markdown(
            """
            <div class="tips-box">
                <strong>💡 Tips for best accuracy:</strong><br>
                • Keep your hand <strong>centered</strong> in the frame.<br>
                • Ensure <strong>good lighting</strong> (avoid dark rooms).<br>
                • Keep a <strong>steady hand</strong> for 1 second.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_out:
        st.markdown("### ✨ AI Result")

        if st.session_state.get("active") and st.session_state.get("live_nohands"):
            st.warning("No hands detected. Show your hand to start translating.")

        result = st.session_state.get("last_good_result")
        job_id = (result or {}).get("job_id")
        already_sent = bool(job_id) and (job_id in st.session_state.feedback_sent_for)

        conf = float((result or {}).get("confidence", 0) or 0)
        gesture = ((result or {}).get("gesture") or "").strip().upper()

        # ✅ Allow feedback whenever job_id exists and it's not NO_HANDS
        can_feedback = bool(job_id) and (gesture not in NO_HANDS_TOKENS)
        disable_feedback = (not can_feedback) or already_sent

        if result and conf < 0.70:
            st.caption("Tip: feedback saves best when confidence ≥ 70% and your hand is centered + steady.")

        debug_box = st.empty()

        if result:
            translation = result.get("translation", "...")
            st.markdown(
                f"""
                <div class="hero-card">
                    <p style="color: #6366f1; font-size: 0.8rem; font-weight: 800; text-transform: uppercase; margin-bottom: 5px;">Detected Translation</p>
                    <h1 class="hero-text">{translation}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(f"**AI Confidence:** {conf:.1%}")
            st.progress(conf)

            m1, m2 = st.columns(2)
            m1.metric("Gesture", result.get("gesture", "—"))
            m2.metric("Latency", f"{result.get('processing_time_ms', 0):.0f}ms")

            st.divider()

            f1, f2 = st.columns(2)

            if f1.button("👍 Correct", use_container_width=True, disabled=disable_feedback):
                try:
                    send_feedback(job_id, accepted=True)
                    st.session_state.feedback_sent_for.add(job_id)
                    st.toast("Saved 👍 Thanks!")
                    # ✅ Production-friendly message
                    debug_box.success("Feedback saved.")
                except Exception as e:
                    msg = str(e)
                    if "No landmarks available for training" in msg:
                        debug_box.warning(
                            "Feedback not saved for training (no hand landmarks detected in that frame). "
                            "Try better lighting + keep your hand centered for 1 second, then click again."
                        )
                    else:
                        debug_box.error(f"Feedback failed: {e}")

            if f2.button("👎 Incorrect", use_container_width=True, disabled=disable_feedback):
                st.session_state.show_correction = True

            if st.session_state.get("show_correction") and job_id and (not disable_feedback):
                corrected = st.selectbox(
                    "Correct gesture:",
                    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    key="corrected_gesture_select",
                )
                if st.button("Submit correction", use_container_width=True):
                    try:
                        send_feedback(job_id, accepted=False, corrected_gesture=corrected)
                        st.session_state.feedback_sent_for.add(job_id)
                        st.toast("Saved 👎 Correction submitted!")
                        st.session_state.show_correction = False
                        # ✅ Production-friendly message
                        debug_box.success("Correction saved.")
                    except Exception as e:
                        msg = str(e)
                        if "No landmarks available for training" in msg:
                            debug_box.warning(
                                "Correction not saved for training (no hand landmarks detected in that frame). "
                                "Try better lighting + keep your hand centered for 1 second, then retry."
                            )
                        else:
                            debug_box.error(f"Feedback failed: {e}")

            if already_sent:
                st.caption("✅ Feedback already saved for this prediction.")
        else:
            st.info("Awaiting gesture input...")

# -------------------- Monitoring Page --------------------
elif page == "Monitoring":
    st.title("📊 System Analytics")
    st.markdown("Real-time performance metrics from the API backend.")

    try:
        api_stats = requests.get(f"{API_URL}/api/stats", timeout=5).json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Hits", api_stats.get("total_predictions", 0))
        c2.metric("Avg Conf", f"{api_stats.get('avg_confidence', 0):.1%}")
        c3.metric("Avg Latency", f"{api_stats.get('avg_latency_ms', 0):.0f}ms")

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Gesture Distribution")
            gd = api_stats.get("gesture_distribution", {})
            if gd:
                st.bar_chart(gd)
            else:
                st.write("No data yet.")

        with col_right:
            st.subheader("Confidence Spread")
            cd = api_stats.get("confidence_distribution", {})
            if cd:
                st.line_chart(cd)
            else:
                st.info("More data needed for trend analysis.")

    except Exception as e:
        st.error(f"Monitoring API unavailable: {e}")