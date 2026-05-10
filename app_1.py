import streamlit as st
import re

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Emosi Livin",
    layout="centered"
)

st.title("📊 Analisis Emosi Nasabah Livin")

st.markdown(
    "Prototype Analisis Emosi dan Sarkasme"
)

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
# EMOTION PREDICTION
# =====================================================

def predict_emotion(text):

    text = clean_text(text)

    # =====================================================
    # SARCASM
    # =====================================================

    if (
        ("bagus" in text or "mantap" in text)
        and
        ("gagal" in text or "error" in text)
    ):

        return "frustrasi", 0.95

    # =====================================================
    # NEGATIVE
    # =====================================================

    negative_words = [
        "gagal",
        "error",
        "lemot",
        "kecewa",
        "maintenance"
    ]

    if any(word in text for word in negative_words):

        return "marah", 0.90

    # =====================================================
    # FEAR
    # =====================================================

    fear_words = [
        "takut",
        "cemas",
        "khawatir"
    ]

    if any(word in text for word in fear_words):

        return "cemas", 0.88

    # =====================================================
    # POSITIVE
    # =====================================================

    positive_words = [
        "bagus",
        "cepat",
        "mantap",
        "membantu",
        "keren"
    ]

    if any(word in text for word in positive_words):

        return "senang", 0.92

    # =====================================================
    # DEFAULT
    # =====================================================

    return "netral", 0.80

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
