"""
lung_chatbot_streamlit.py
Single-file Streamlit demo with robust Grad-CAM visualization + Google Drive model fetch.

- Downloads model from Google Drive if not present locally.
- Verifies HDF5 signature before attempting to load.
- Shows download progress in UI.
- Keeps original features: upload .jpg/.npy, inference, Grad-CAM, Kannada chatbot + TTS.
"""

import os
import io
import re
import time
import stat
import tempfile
import traceback
import requests

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from gtts import gTTS

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

# ---------- CONFIG ----------
MODEL_PATH = r"resnet50_lung_cancer.h5"  # local target path (change if you want)
INPUT_SIZE = (224, 224)
CLASS_MAP = {0: "Normal", 1: "Benign", 2: "Malignant"}

# Google Drive file id for the .h5 model (replace with yours if needed)
GOOGLE_DRIVE_ID = "1QN6jTC-pZYXmazzA-vMcnDksLORKFbUA"
DRIVE_BASE_URL = "https://drive.google.com/uc?export=download&id="
# ----------------------------

# ----------------- Utilities for downloading and verifying -----------------
def _extract_confirm_token_from_html(html_text: str):
    patterns = [
        r'confirm=([0-9A-Za-z_-]+)&amp;id=',
        r'confirm=([0-9A-Za-z_-]+)&',
        r'download_warning[0-9]+=([0-9A-Za-z_-]+)',
        r'confirm=([0-9A-Za-z_-]+)\"',
        r'name="confirm" value="([0-9A-Za-z_-]+)"'
    ]
    for p in patterns:
        m = re.search(p, html_text)
        if m:
            return m.group(1)
    return None

def _looks_like_hdf5(path):
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig == b'\x89HDF\r\n\x1a\n'
    except Exception:
        return False

def download_from_google_drive_strict(file_id: str, out_path: str, progress_cb=None, timeout=30):
    """
    Robustly download a file from Google Drive, handling confirm tokens for large files.
    Verifies file size and HDF5 signature; returns (out_path, None) on success or (None, error_message).
    progress_cb(percent_int) optional callback for progress updates.
    """
    url = DRIVE_BASE_URL + file_id
    session = requests.Session()

    try:
        resp = session.get(url, stream=True, timeout=timeout)
    except Exception as e:
        return None, f"Initial request failed: {e}"

    # read a small prefix to check if HTML returned
    start = b""
    try:
        # resp.raw is available when stream=True
        start = resp.raw.read(2048)
    except Exception:
        start = resp.content[:2048]

    text_start = ""
    try:
        text_start = start.decode("utf-8", errors="ignore")
    except Exception:
        text_start = ""

    # token from cookies
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token is None:
        token = _extract_confirm_token_from_html(text_start)

    if token:
        try:
            resp = session.get(url + "&confirm=" + token, stream=True, timeout=timeout)
        except Exception as e:
            return None, f"Request with confirm token failed: {e}"

    # Save to temp .part file while tracking progress
    tmp = out_path + ".part"
    try:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None
        chunk_size = 32768
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total:
                        pct = int(downloaded * 100 / total)
                        progress_cb(min(pct, 100))
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return None, f"Failed while writing file: {e}"

    # sanity size check
    try:
        size = os.path.getsize(tmp)
    except Exception:
        size = 0

    if size < 1024:
        # read content for debug
        try:
            with open(tmp, "rb") as f:
                sample = f.read(4096).decode("utf-8", errors="ignore")
        except Exception:
            sample = "<couldn't read partial file>"
        try:
            os.remove(tmp)
        except Exception:
            pass
        return None, f"Downloaded file too small ({size} bytes). Likely an HTML page. Leading content:\n\n{sample[:1000]}"

    # move into place
    try:
        os.replace(tmp, out_path)
    except Exception as e:
        return None, f"Failed to rename temp file: {e}"

    # chmod
    try:
        os.chmod(out_path, stat.S_IREAD | stat.S_IWRITE)
    except Exception:
        pass

    # verify HDF5 signature
    if not _looks_like_hdf5(out_path):
        try:
            with open(out_path, "rb") as f:
                sample = f.read(2048).decode("utf-8", errors="ignore")
        except Exception:
            sample = "<couldn't read file for debug>"
        # keep file (user may want to inspect), but report error
        return None, ("Downloaded file does not start with HDF5 signature. "
                      f"File size: {size} bytes. Leading content (first 1000 chars):\n\n{sample[:1000]}\n\n"
                      "This usually means Google Drive returned an HTML page (permission/preview) or the file is corrupted. "
                      "Ensure the Drive file is the binary .h5 and 'Anyone with the link' is enabled.")
    return out_path, None

# ---------- Model presence & load flow ----------
def ensure_model_present_ui(file_id: str, target_path: str):
    """
    Ensures model exists locally. If missing, downloads with progress and returns (True, message) or (False, error).
    This function interacts with Streamlit UI (progress bar / info messages).
    """
    if os.path.exists(target_path):
        return True, f"Model already exists at: {target_path}"

    st.info("Model not found locally. Attempting download from Google Drive...")
    progress = st.progress(0)
    status_text = st.empty()

    def progress_cb(pct):
        try:
            progress.progress(pct)
            status_text.info(f"Downloading model: {pct}%")
        except Exception:
            pass

    out, err = download_from_google_drive_strict(file_id, target_path, progress_cb=progress_cb)
    if err:
        progress.empty()
        status_text.error("Download failed.")
        return False, err
    progress.progress(100)
    status_text.success("Download complete.")
    return True, f"Downloaded model to {out}"

@st.cache_resource(show_spinner=False)
def load_keras_model_cached(path):
    """
    Load the Keras model from disk (cached to avoid reloading repeatedly).
    Return (model_or_none, message)
    """
    if not os.path.exists(path):
        return None, f"Model not found at: {path}"
    try:
        model = keras_load_model(path)
        return model, f"Loaded model from: {path}"
    except Exception as e:
        return None, f"Error loading model: {e}"

# ----------------- Preprocessing, GradCAM, overlay, inference (unchanged) -----------------
def preprocess_slice(slice_img, target_size=INPUT_SIZE):
    img = slice_img.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img.astype(np.float32)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        name = layer.name.lower()
        if 'conv' in name:
            return layer.name
        if layer.__class__.__name__.lower().startswith('conv'):
            return layer.name
    raise ValueError("No convolutional layer found in model to compute Grad-CAM.")

def make_gradcam_heatmap(img_array, model, pred_index=None, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        predictions = tf.convert_to_tensor(predictions)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    weighted = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
    heatmap /= max_val
    return heatmap.numpy().astype(np.float32)

def overlay_heatmap_on_image(orig_rgb, heatmap, thresh=0.35):
    if orig_rgb.dtype != np.uint8:
        orig = (255 * (orig_rgb - orig_rgb.min()) / (orig_rgb.ptp() + 1e-8)).astype(np.uint8)
    else:
        orig = orig_rgb.copy()

    hmap_u8 = (heatmap * 255).astype(np.uint8)
    hmap_resized = cv2.resize(hmap_u8, (orig.shape[1], orig.shape[0]))
    heatmap_color = cv2.applyColorMap(hmap_resized, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    _, mask = cv2.threshold(hmap_resized, int(thresh * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 50:
            continue
        cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb, annotated_rgb

def run_inference_with_gradcam(model, volume_array, return_gradcam=True):
    try:
        if model is None:
            mean_val = float(np.nanmean(volume_array))
            score = 1.0 / (1.0 + np.exp(-0.01*(mean_val-100)))
            label = "ಸ್ಥೂಲ ಸಂಶಯ (Malignant)" if score > 0.5 else "ಸಂದೇಹದಿಲ್ಲ (Benign)"
            return {"label": label, "score": float(score), "probs": None, "notes": "Demo inference; replace with real model.", "gradcam": None}

        arr = np.squeeze(volume_array)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        S = arr.shape[0]

        preproc_list = []
        orig_resized = []
        for i in range(S):
            sl = arr[i]
            if sl.ndim == 2:
                rgb = np.stack([sl, sl, sl], axis=-1)
            else:
                rgb = sl
            rgb_resized = cv2.resize(rgb.astype(np.uint8), INPUT_SIZE)
            orig_resized.append(rgb_resized)
            x = preprocess_slice(sl, target_size=INPUT_SIZE)
            preproc_list.append(x)

        X = np.stack(preproc_list, axis=0).astype(np.float32)

        preds = model.predict(X, batch_size=16)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)
        if preds.ndim != 2:
            return {"error": f"Unexpected prediction shape: {preds.shape}", "trace": None, "gradcam": None}
        row_sums = preds.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            preds = tf.nn.softmax(preds, axis=-1).numpy()

        probs = preds
        mean_probs = probs.mean(axis=0)
        class_idx = int(np.argmax(mean_probs))
        score = float(mean_probs[class_idx])

        label_en = CLASS_MAP.get(class_idx, f"class_{class_idx}")
        kn_map = {"Normal": "ಸಾಮಾನ್ಯ (Normal)", "Benign": "ಸಂದೇಹವಿಲ್ಲ (Benign)", "Malignant": "ಸ್ಥೂಲ ಸಂಶಯ (Malignant)"}
        label_kn = kn_map.get(label_en, label_en)

        gradcam_info = None
        if return_gradcam:
            per_slice_scores = probs[:, class_idx]
            rep_idx = int(np.argmax(per_slice_scores))
            if rep_idx < 0 or rep_idx >= len(preproc_list):
                rep_idx = 0
            single_input = np.expand_dims(preproc_list[rep_idx], axis=0).astype(np.float32)
            try:
                last_conv = None
                try:
                    last_conv = find_last_conv_layer(model)
                except Exception as e:
                    st.info(f"Last conv detection: {e}")
                    last_conv = None
                heatmap = make_gradcam_heatmap(single_input, model, pred_index=class_idx, last_conv_layer_name=last_conv)
                if heatmap is None or heatmap.size == 0:
                    raise RuntimeError("Heatmap empty")
                overlay_rgb, annotated_rgb = overlay_heatmap_on_image(orig_resized[rep_idx], heatmap, thresh=0.35)
                gradcam_info = {"slice_index": rep_idx, "overlay_rgb": overlay_rgb, "annotated_rgb": annotated_rgb}
            except Exception as e:
                gradcam_info = {"error": str(e), "trace": traceback.format_exc()}

        return {"label": label_kn, "score": score, "probs": mean_probs.tolist(), "notes": "Model-based inference (mean over slices).", "gradcam": gradcam_info}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc(), "gradcam": None}

# ----------------- Kannada chatbot / TTS -----------------
KANNADA_TEMPLATES = {
    "greeting": "ನಮಸ್ತೆ! 나는 ನಿಮ್ಮ ಲಂಗ್-ಕ್ಯಾನ್ಸರ್ ಪ್ರೋಟೋಟೈಪ್ ಚಾಟ್‌ಬಾಟ್. ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "inference_result": "ಮೌಲ್ಯಮಾಪನ ಫಲಿತಾಂಶ:\n\nಫಲಿತಾಂಶ: {label}\nconfidence (ಸ್ಕೋರ್): {score:.3f}\nನೋಟ್: {notes}",
    "explain_how": "ಈ ಮಾಡೆಲ್ 3D CT ವಾಲ್ಯೂಮ್ ಗಳನ್ನು ಒಳಗೆ ತೆಗೆದುಕೊಂಡು ಕಂಡುಬರುವ ಗುಣಲಕ್ಷಣಗಳನ್ನು ಆಧರಿಸಿ ಅನುಮಾನಿತ ತಂತು/ನೋಡ್ ಗಳನ್ನು ಗುರುತಿಸುತ್ತದೆ.",
    "accuracy": "ಮಾಡೆಲ್‌ಗಾಗಿ ಉದಾಹರಣೆ accuracy = {acc:.2f}.",
    "limitations": "ಸೀಮಿತತೆ: ಇದು ಡೆಮೊ; ವೈದ್ಯ ಸಲಹೆ ಅವಶ್ಯಕ.",
    "thanks": "ಧನ್ಯವಾದಗಳು!"
}

def detect_intent(user_text):
    t = user_text.strip().lower()
    if any(x in t for x in ["hello","hi","namaste","ನಮಸ್ತೆ","ಹಲೋ","hey"]):
        return "greeting"
    if any(x in t for x in ["predict","inference","run inference","result","ಫಲ","ಫಲಿತಾಂಶ"]):
        return "predict"
    if any(x in t for x in ["how","work","ಹೇಗೆ","ಮಾಡುತ್ತದೆ","explain","ವಿವರಣೆ"]):
        return "explain"
    if any(x in t for x in ["accuracy","precision","sensitivity","acc"]):
        return "accuracy"
    if any(x in t for x in ["limitations","limits","ಸೀಮಿತತೆ"]):
        return "limitations"
    if any(x in t for x in ["thanks","thank","ಧನ್ಯ","bye"]):
        return "thanks"
    return "unknown"

def answer_in_kannada(intent, context=None):
    if intent == "greeting":
        return KANNADA_TEMPLATES["greeting"]
    if intent == "explain":
        return KANNADA_TEMPLATES["explain_how"]
    if intent == "accuracy":
        acc = context.get("acc", 0.85) if context else 0.85
        return KANNADA_TEMPLATES["accuracy"].format(acc=acc)
    if intent == "limitations":
        return KANNADA_TEMPLATES["limitations"]
    if intent == "thanks":
        return KANNADA_TEMPLATES["thanks"]
    if intent == "predict":
        return "ದಯವಿಟ್ಟು JPG ಸ್ಲೈಸುಗಳನ್ನು ಅಥವಾ .npy ಫೈಲ್ ಅನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು 'Run Inference' ಒತ್ತಿ."
    return "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ."

def tts_kannada(text):
    try:
        tts = gTTS(text=text, lang='kn')
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        st.error("TTS error: " + str(e))
        return None

# ----------------- Streamlit UI -----------------
st.title("Lung Cancer Detector")
st.markdown("*DISCLAIMER:* This is a demo prototype. Not a medical diagnosis tool.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Model / Inference")

    # Ensure model is present (downloads if missing)
    ok, msg = ensure_model_present_ui(GOOGLE_DRIVE_ID, MODEL_PATH)
    if not ok:
        st.error("Model download/verification failed:")
        st.code(msg)
        st.info("Make sure the Drive file is the binary .h5 and sharing is set to 'Anyone with the link → Viewer'.")
        model = None
    else:
        st.success(msg)
        model, load_msg = load_keras_model_cached(MODEL_PATH)
        if model is None:
            st.error("Failed to load model:")
            st.code(load_msg)
        else:
            st.success(load_msg)

    uploaded_files = st.file_uploader("Upload CT scan slices (.jpg) or a single .npy", type=["jpg","jpeg","npy"], accept_multiple_files=True)
    arr = None

    if uploaded_files:
        npy_files = [f for f in uploaded_files if f.name.lower().endswith('.npy')]
        jpg_files = [f for f in uploaded_files if f.name.lower().endswith(('.jpg','.jpeg'))]

        if len(npy_files) == 1 and len(uploaded_files) == 1:
            arr = np.load(io.BytesIO(npy_files[0].read()))
            st.write("Loaded .npy volume shape:", arr.shape)
            st.session_state["last_volume"] = arr
        elif len(jpg_files) >= 1:
            jpg_files = sorted(jpg_files, key=lambda x: x.name)
            imgs = []
            for file in jpg_files:
                img = Image.open(file).convert("L")
                imgs.append(np.array(img))
            arr = np.stack(imgs, axis=0)
            st.write("Stacked JPG volume shape:", arr.shape)
            st.session_state["last_volume"] = arr
        else:
            st.warning("Upload either a single .npy file or one/more JPG slices.")
    else:
        arr = st.session_state.get("last_volume", None)

    if st.button("Run Inference"):
        if arr is None:
            st.warning("Please upload JPG slices or a .npy volume first.")
        else:
            with st.spinner("Running inference + Grad-CAM..."):
                result = run_inference_with_gradcam(model, arr, return_gradcam=True)
                if "error" in result:
                    st.error("Inference failed: " + result.get("error", "unknown"))
                    if result.get("trace"):
                        st.text(result.get("trace"))
                else:
                    out_text = KANNADA_TEMPLATES["inference_result"].format(label=result.get("label","N/A"), score=result.get("score",0.0), notes=result.get("notes",""))
                    st.markdown("### ಫಲಿತಾಂಶ (Kannada)")
                    st.code(out_text)

                    audio_file = tts_kannada(out_text)
                    if audio_file:
                        st.audio(open(audio_file, "rb").read(), format="audio/mp3", autoplay=True)
                        st.session_state["last_bot_audio"] = audio_file

                    st.session_state["last_result"] = result

                    gc = result.get("gradcam")
                    if gc is None:
                        st.info("No Grad-CAM (model missing or disabled).")
                    elif isinstance(gc, dict) and "error" in gc:
                        st.error("Grad-CAM generation failed: " + gc.get("error", ""))
                        if gc.get("trace"):
                            st.text(gc.get("trace"))
                    elif isinstance(gc, dict) and "overlay_rgb" in gc:
                        rep_idx = gc.get("slice_index", None)
                        st.markdown(f"**Representative slice index:** {rep_idx}")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(gc["annotated_rgb"], caption="Original slice with bounding box (red)", use_container_width=True)
                        with c2:
                            st.image(gc["overlay_rgb"], caption="Grad-CAM overlay (heatmap)", use_container_width=True)
                    else:
                        st.info("Grad-CAM: unexpected format; see logs.")

with col2:
    st.header("Chat (Kannada)")
    st.markdown("Type questions in Kannada or English; the bot replies in Kannada (with speech).")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Your question (Kannada/English):", key="user_input")
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please type a question.")
        else:
            st.session_state["chat_history"].append(("user", user_input))
            intent = detect_intent(user_input)

            if intent == "predict":
                if "last_result" in st.session_state:
                    r = st.session_state["last_result"]
                    bot_text = KANNADA_TEMPLATES["inference_result"].format(label=r.get("label","N/A"), score=r.get("score",0.0), notes=r.get("notes",""))
                else:
                    bot_text = answer_in_kannada(intent)
            elif intent == "accuracy":
                bot_text = answer_in_kannada(intent, context={"acc":0.92})
            else:
                bot_text = answer_in_kannada(intent)

            st.session_state["chat_history"].append(("bot", bot_text))
            audio_path = tts_kannada(bot_text)
            if audio_path:
                st.audio(open(audio_path, "rb").read(), format="audio/mp3", autoplay=True)
                st.session_state["last_bot_audio"] = audio_path

    for speaker, text in st.session_state["chat_history"][-20:]:
        if speaker == "user":
            st.markdown(f"*You:* {text}")
        else:
            st.markdown(f"*Bot:* {text}")
