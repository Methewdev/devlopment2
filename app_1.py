import streamlit as st
import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Emosi Livin",
    layout="centered"
)

st.title("📊 Analisis Emosi Nasabah Livin")

st.markdown(
    "Deteksi Emosi berbasis IndoBERT sesuai proposal tesis"
)

# =====================================================
# MODEL CONFIG
# =====================================================

MODEL_NAME = "indobenchmark/indobert-base-p1"

emotion_classes = [
    "cemas",
    "frustrasi",
    "marah",
    "netral",
    "puas",
    "senang"
]

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6
    )

    return tokenizer, model

tokenizer, model = load_model()

# =====================================================
# CLEANING
# =====================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s!?]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# =====================================================
# IMPLICIT SARCASM HANDLING
# =====================================================

positive_words = [
    "bagus",
    "mantap",
    "hebat",
    "keren",
    "terbaik"
]

negative_words = [
    "error",
    "gagal",
    "maintenance",
    "lemot",
    "pending"
]

def sarcasm_boost(text, emotion):

    pos_found = any(
        word in text for word in positive_words
    )

    neg_found = any(
        word in text for word in negative_words
    )

    if pos_found and neg_found:

        if emotion == "senang":

            return "frustrasi"

    return emotion

# =====================================================
# PREDICTION
# =====================================================

def predict_emotion(text):

    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probs,
        dim=1
    ).item()

    confidence = probs[0][prediction].item()

    emotion = emotion_classes[prediction]

    # implicit sarcasm handling
    emotion = sarcasm_boost(
        cleaned,
        emotion
    )

    return emotion, confidence

# =====================================================
# INPUT
# =====================================================

text = st.text_area(
    "Masukkan ulasan nasabah"
)

# =====================================================
# BUTTON
# =====================================================

if st.button("Analisis"):

    if text.strip() == "":

        st.warning(
            "Masukkan ulasan terlebih dahulu"
        )

    else:

        emotion, confidence = predict_emotion(
            text
        )

        emotion_colors = {
            "marah": "red",
            "frustrasi": "orange",
            "cemas": "purple",
            "senang": "green",
            "puas": "blue",
            "netral": "gray"
        }

        color = emotion_colors.get(
            emotion,
            "black"
        )

        st.markdown("## Hasil Analisis")

        st.markdown(
            f"""
            <h2 style='color:{color};'>
            {emotion.upper()}
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

# =====================================================
# EXAMPLE
# =====================================================

st.markdown("---")

st.subheader("Contoh Sarkasme")

samples = [
    "Bagus banget aplikasinya transfer gagal terus",
    "Mantap maintenance tiap malam",
    "Terima kasih login gagal"
]

for s in samples:

    st.code(s)
