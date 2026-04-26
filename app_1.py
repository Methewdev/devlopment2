import os
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans

st.set_page_config(page_title="Analisis Emosi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# LOAD MODEL (SAFE)
# =========================
def load_model_safe():
    PRIMARY_MODEL = "envidevelopment/sentiment-banking"
    FALLBACK_MODEL = "indobenchmark/indobert-base-p1"

    # Try primary
    try:
        tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(PRIMARY_MODEL)
        return tokenizer, model, "primary"
    except:
        pass

    # Try fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL,
            num_labels=6
        )
        return tokenizer, model, "fallback"
    except:
        return None, None, "error"

# =========================
# LOAD ON DEMAND
# =========================
@st.cache_resource
def get_model():
    tokenizer, model, status = load_model_safe()

    if status == "error":
        return None, None, status

    model.to("cpu")
    model.eval()

    return tokenizer, model, status

# =========================
# UI LOAD BUTTON (IMPORTANT)
# =========================
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    if st.button("🔄 Load Model"):
        tokenizer, model, status = get_model()

        if status == "error":
            st.error("❌ Semua model gagal diload")
            st.stop()

        if status == "fallback":
            st.warning("⚠️ Menggunakan model fallback")

        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.model_loaded = True

        st.success("✅ Model berhasil dimuat")
        st.rerun()

    st.stop()

# =========================
# MODEL READY
# =========================
tokenizer = st.session_state.tokenizer
model = st.session_state.model

# Label aman
if hasattr(model.config, "id2label"):
    id2label = model.config.id2label
else:
    id2label = {
        0: "senang",
        1: "marah",
        2: "kecewa",
        3: "takut",
        4: "netral",
        5: "frustrasi"
    }

# =========================
# PREDICT
# =========================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# INPUT TEKS
# =========================
if menu == "Input Teks":
    text = st.text_area("Masukkan teks")

    if st.button("Analisis"):
        if text.strip():
            result = predict(text)
            st.json(result)
        else:
            st.warning("Isi teks dulu")

# =========================
# UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    file = st.file_uploader("Upload CSV (kolom: text)")

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("Kolom 'text' tidak ditemukan")
            st.stop()

        if st.button("Proses"):
            results = df["text"].astype(str).apply(predict)
            emotion_df = pd.DataFrame(list(results))

            df = pd.concat([df, emotion_df], axis=1)

            kmeans = KMeans(n_clusters=3, random_state=42)
            df["cluster"] = kmeans.fit_predict(emotion_df)

            st.dataframe(df.head())
