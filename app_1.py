import streamlit as st
import re

from transformers import pipeline

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Emosi Livin",
    layout="centered"
)

st.title("📊 Analisis Emosi Nasabah Livin")

st.markdown(
    "Analisis Emosi berbasis Transformer sesuai proposal tesis"
)

# =====================================================
# LOAD PIPELINE
# =====================================================

@st.cache_resource
def load_pipeline():

    classifier = pipeline(
        "text-classification",
        model="indobenchmark/indobert-base-p1"
    )

    return classifier

classifier = load_pipeline()

# =====================================================
# EMOTION LABEL
# =====================================================

emotion_classes = [
    "cemas",
    "frustrasi",
    "marah",
    "netral",
    "puas",
    "senang"
]

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
# SARCASM HANDLING
# =====================================================

positive_words = [
    "bagus",
    "mantap",
    "hebat",
    "keren"
]

negative_words = [
    "error",
    "gagal",
    "maintenance",
    "lemot"
]

def detect_sarcasm(text):

    pos_found = any(
        word in text for word in positive_words
    )

    neg_found = any(
        word in text for word in negative_words
    )

    return pos_found and neg_found

# =====================================================
# PREDICT
# =====================================================

def predict_emotion(text):

    cleaned = clean_text(text)

    result = classifier(cleaned)

    score = result[0]["score"]

    label_id = int(
        result[0]["label"].split("_")[-1]
    )

    emotion = emotion_classes[
        label_id % len(emotion_classes)
    ]

    if detect_sarcasm(cleaned):

        emotion = "frustrasi"

    return emotion, score

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
            "Masukkan teks terlebih dahulu"
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
            "Confidence",
            f"{confidence*100:.2f}%"
        )

# =====================================================
# SAMPLE
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
