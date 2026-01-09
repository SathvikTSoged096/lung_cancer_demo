"""
Lung cancer Streamlit demo (fixed & cleaned)
This is a demo prototype and NOT a medical tool.
"""

# -------------------- IMPORTS --------------------
import os
import io
import tempfile
import traceback
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from gtts import gTTS

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

from auth_db import init_db, create_user, verify_user
# ------------------------------------------------

# -------------------- CONFIG ---------------------
st.set_page_config(
    page_title="Lung Cancer Detector",
    layout="wide"
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_ID = "1QN6jTC-pZYXmazzA-vMcnDksLORKFbUA"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}&export=download"
MODEL_PATH = "resnet50_lung_cancer.h5"

INPUT_SIZE = (224, 224)
CLASS_MAP = {0: "Normal", 1: "Benign", 2: "Malignant"}
# ------------------------------------------------

# -------------------- INIT -----------------------
init_db()

if "page" not in st.session_state:
    st.session_state.page = "login"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# ------------------------------------------------

# -------------------- HIDE STREAMLIT UI ----------
st.markdown(
    """
    <style>
    header, footer, #MainMenu {display:none !important;}
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stDecoration"] {display:none !important;}
    .block-container {padding-bottom:0 !important;}
    </style>
    """,
    unsafe_allow_html=True
)
# ------------------------------------------------

# -------------------- LOGIN ----------------------
if st.session_state.page == "login":
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        ok, name, role = verify_user(u, p)
        if ok:
            st.session_state.authenticated = True
            st.session_state.name = name
            st.session_state.role = role
            st.session_state.page = "app"
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("🆕 New user? Sign up"):
        st.session_state.page = "signup"
        st.rerun()

    st.stop()
# ------------------------------------------------

# -------------------- SIGNUP ---------------------
if st.session_state.page == "signup":
    st.title("🆕 Create Account")

    nu = st.text_input("Username")
    nn = st.text_input("Full Name")
    ne = st.text_input("Email")
    npw = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Register"):
            try:
                create_user(nu, nn, ne, npw)
                st.success("Registration successful! Please log in.")
                st.session_state.page = "login"
                st.rerun()
            except Exception:
                st.error("Username already exists")

    with col2:
        if st.button("Cancel"):
            st.session_state.page = "login"
            st.rerun()

    st.stop()
# ------------------------------------------------

# -------------------- AUTH GUARD -----------------
if not st.session_state.authenticated:
    st.stop()
# ------------------------------------------------

# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Welcome {st.session_state.name}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.page = "login"
    st.rerun()
# ------------------------------------------------

# -------------------- HELPERS --------------------
def _looks_like_html(path):
    try:
        with open(path, "rb") as f:
            return b"<html" in f.read(1024).lower()
    except:
        return True


@st.cache_resource
def load_keras_model(path):
    if not os.path.exists(path) or _looks_like_html(path):
        gdown.download(MODEL_URL, path, quiet=False)
    return keras_load_model(path, compile=False)


def preprocess_slice(img):
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = cv2.resize(img, INPUT_SIZE) / 255.0
    return img.astype(np.float32)


def generate_patient_report(name, label, score):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)

    w, h = A4
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w/2, h-2*cm, "Lung Cancer Detection Report")
    c.setFont("Helvetica", 12)

    y = h - 4*cm
    c.drawString(2*cm, y, f"Patient Name: {name}"); y -= cm
    c.drawString(2*cm, y, f"Date: {datetime.now()}"); y -= cm
    c.drawString(2*cm, y, f"Result: {label}"); y -= cm
    c.drawString(2*cm, y, f"Confidence: {score:.3f}")

    c.showPage()
    c.save()
    return tmp.name
# ------------------------------------------------

# -------------------- UI -------------------------
st.title("🫁 Lung Cancer Detector")
st.markdown("*DISCLAIMER: Demo only. Not a medical diagnosis tool.*")

col1, col2 = st.columns(2)

with col1:
    st.header("Model / Inference")

    model = load_keras_model(MODEL_PATH)

    files = st.file_uploader(
        "Upload CT slices (.jpg) or volume (.npy)",
        type=["jpg", "jpeg", "npy"],
        accept_multiple_files=True
    )

    volume = None
    if files:
        if files[0].name.endswith(".npy"):
            volume = np.load(io.BytesIO(files[0].read()), allow_pickle=True)
        else:
            imgs = [np.array(Image.open(f).convert("L")) for f in files]
            volume = np.stack(imgs)

    if st.button("Run Inference") and volume is not None:
        X = np.stack([preprocess_slice(s) for s in volume])
        preds = model.predict(X)
        probs = preds.mean(axis=0)

        idx = int(np.argmax(probs))
        label = CLASS_MAP[idx]
        score = float(probs[idx])

        st.session_state.last_result = {
            "label": label,
            "score": score
        }

        st.success(f"Result: {label} ({score:.3f})")

with col2:
    st.header("🧾 Patient Report")

    pname = st.text_input("Patient Name")

    if st.button("Generate PDF"):
        if "last_result" not in st.session_state:
            st.warning("Run inference first")
        elif pname.strip() == "":
            st.warning("Enter patient name")
        else:
            r = st.session_state.last_result
            pdf = generate_patient_report(pname, r["label"], r["score"])
            with open(pdf, "rb") as f:
                st.download_button(
                    "Download Report",
                    f,
                    file_name=f"{pname}_report.pdf",
                    mime="application/pdf"
                )
# ------------------------------------------------
