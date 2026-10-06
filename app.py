import streamlit as st
import torch
import av
import cv2

from PIL import Image

from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification
)

from streamlit_webrtc import webrtc_streamer


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Real-Time Sign Language",
    page_icon="🤟",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(" Real-Time Sign Language Recognition")

st.write(
    "Alphabet Sign Language Detection using SigLIP"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

st.sidebar.write(
    f"Device: `{DEVICE}`"
)

if torch.cuda.is_available():

    st.sidebar.success(
        torch.cuda.get_device_name(0)
    )

else:

    st.sidebar.warning(
        "Running on CPU"
    )


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "prithivMLmods/"
    "Alphabet-Sign-Language-Detection"
)


@st.cache_resource
def load_model():

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = SiglipForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


processor, model = load_model()


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z"
]


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    image = Image.fromarray(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0
    )

    return (
        LABELS[index.item()],
        confidence.item()
    )


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    # Resize
    img = cv2.resize(
        img,
        (640, 480)
    )

    try:

        letter, confidence = predict(img)

    except Exception as e:

        print("Prediction error:", e)

        letter = "?"
        confidence = 0.0

    # ========================================================
    # DRAW UI
    # ========================================================

    cv2.rectangle(
        img,
        (15, 15),
        (350, 135),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        f"Sign: {letter}",
        (30, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        img,
        f"Confidence: {confidence * 100:.1f}%",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# ============================================================
# CAMERA
# ============================================================

st.subheader("📷 Live Camera")

webrtc_streamer(
    key="sign-language",

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    }
)


# ============================================================
# INFO
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Classes",
        "26"
    )

with col2:
    st.metric(
        "Model",
        "SigLIP"
    )

with col3:
    st.metric(
        "Task",
        "A-Z Detection"
    )