# ui/app.py
import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import os

# config
API_URL = os.getenv("API_URL", "http://api:8000")
CLASS_LABELS = {
    'nv'   : 'Melanocytic Nevi (Mole)',
    'mel'  : 'Melanoma',
    'bkl'  : 'Benign Keratosis',
    'bcc'  : 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc' : 'Vascular Lesion',
    'df'   : 'Dermatofibroma',
}
RISK_LEVEL = {
    'nv'   : ('Low',      '🟢'),
    'mel'  : ('High',     '🔴'),
    'bkl'  : ('Low',      '🟢'),
    'bcc'  : ('Medium',   '🟡'),
    'akiec': ('Medium',   '🟡'),
    'vasc' : ('Low',      '🟢'),
    'df'   : ('Low',      '🟢'),
    'unknown': ('Unknown','⚪'),
}

# page config
st.set_page_config(
    page_title = "Skin Disease Classifier",
    page_icon  = "🔬",
    layout     = "wide"
)

# header
st.title("🔬 Skin Disease Classifier")
st.markdown(
    "AI-powered skin lesion classification using "
    "**EfficientNet-B4** trained on HAM10000 dataset"
)
st.divider()

# check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"API Status: {health['status'].upper()} ✅")
    st.sidebar.info(f"Model: {health['model']}")
    st.sidebar.info(f"Device: {health['device']}")
    st.sidebar.info(f"Classes: {health['classes']}")
except Exception:
    st.sidebar.error("API is offline ❌ — start FastAPI first")

# sidebar info
st.sidebar.title("About")
st.sidebar.markdown("""
This app classifies skin lesions into 7 categories:
- 🟢 **nv** — Melanocytic Nevi
- 🔴 **mel** — Melanoma
- 🟢 **bkl** — Benign Keratosis
- 🟡 **bcc** — Basal Cell Carcinoma
- 🟡 **akiec** — Actinic Keratosis
- 🟢 **vasc** — Vascular Lesion
- 🟢 **df** — Dermatofibroma
""")
st.sidebar.warning(
    "⚠️ This tool is for educational purposes only. "
    "Always consult a dermatologist for medical advice."
)

# main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Prediction Results")

    if uploaded_file:
        with st.spinner("Analyzing image..."):
            # call FastAPI
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)

            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("image.jpg", buf, "image/jpeg")},
                    timeout=30
                )
                result = response.json()
                status = result.get("status", "success")  # NEW

                # ── NEW: handle gatekeeper rejections first ────────────
                if status == "rejected_human":
                    st.warning(f"🧑 {result['message']}")
                    st.caption(
                        "Tip: take a close-up photo of just the affected "
                        "skin patch, not your face or full body."
                    )

                elif status == "rejected_invalid":
                    st.warning(f"📷 {result['message']}")
                    st.caption(
                        "Tip: upload a clear, close-up photo of the "
                        "skin lesion you'd like analyzed."
                    )

                # ── EXISTING: unknown / low-confidence skin image ───────
                elif result["is_unknown"]:
                    st.error("⚪ Unknown — not a recognized skin lesion")
                    if result.get("confidence") is not None:
                        st.metric("Confidence", f"{result['confidence']:.1%}")

                # ── EXISTING: successful prediction — unchanged ─────────
                else:
                    cls      = result["predicted_class"]
                    label    = result["label"]
                    conf     = result["confidence"]
                    risk, icon = RISK_LEVEL.get(cls, ('Unknown', '⚪'))

                    # main prediction
                    st.success(f"{icon} **{label}**")
                    st.metric("Confidence", f"{conf:.1%}")
                    st.metric("Risk Level", risk)

                    # probability bar chart
                    probs  = result["all_probabilities"]
                    labels = list(probs.keys())
                    values = [probs[l] * 100 for l in labels]
                    colors = [
                        '#FF4B4B' if l == cls else '#4B8BFF'
                        for l in labels
                    ]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=labels,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in values],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title     = "Class Probabilities",
                        xaxis_title = "Probability (%)",
                        height    = 300,
                        margin    = dict(l=0, r=40, t=40, b=0),
                        showlegend = False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # medical info
                    with st.expander("ℹ️ About this condition"):
                        info = {
                            'nv'   : "Melanocytic nevi are common begining moles. Usually harmless but monitor for changes in size, shape or color.",
                            'mel'  : "Melanoma is the most serious skin cancer. Early detection is critical — please consult a dermatologist immediately.",
                            'bkl'  : "Benign keratosis are non-cancerous skin growths. Generally harmless but can be removed for cosmetic reasons.",
                            'bcc'  : "Basal cell carcinoma is the most common skin cancer. Grows slowly and rarely spreads — highly treatable if caught early.",
                            'akiec': "Actinic keratosis is a precancerous condition caused by sun damage. Treatment recommended to prevent progression.",
                            'vasc' : "Vascular lesions are abnormalities of blood vessels. Usually benign and may not require treatment.",
                            'df'   : "Dermatofibroma is a benign fibrous nodule. Usually harmless and requires no treatment.",
                        }
                        st.info(info.get(cls, ""))

            except Exception as e:
                st.error(f"Error connecting to API: {e}")
    else:
        st.info("👆 Upload an image to get started")

# footer
st.divider()
st.markdown(
    "<center>Built with PyTorch + FastAPI + Streamlit | "
    "HAM10000 Dataset | EfficientNet-B4</center>",
    unsafe_allow_html=True
)