import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Emosi & Sarkasme", layout="wide")
st.title("📊 Analisis Emosi, Sarkasme & Segmentasi Nasabah")

# =========================
# DETEKSI EMOSI
# =========================
def predict_emotion(text):
    text = text.lower()

    if any(k in text for k in ["bagus", "mantap", "cepat", "puas"]):
        return "Senang 😊"
    elif any(k in text for k in ["buruk", "jelek", "error", "gagal"]):
        return "Marah 😡"
    elif any(k in text for k in ["lambat", "kecewa", "tidak sesuai"]):
        return "Kecewa 😞"
    else:
        return "Netral 😐"

# =========================
# DETEKSI SARKASME
# =========================
def detect_sarcasm(text):
    text = text.lower()

    sarcasm_patterns = [
        "mantap sekali padahal",
        "bagus tapi",
        "hebat ya padahal",
        "terima kasih atas",
        "luar biasa buruk",
        "keren banget error",
    ]

    for pattern in sarcasm_patterns:
        if pattern in text:
            return "Sarkasme Terdeteksi ⚠️"

    # kombinasi positif + negatif
    if ("bagus" in text or "mantap" in text) and ("error" in text or "lambat" in text):
        return "Sarkasme Terdeteksi ⚠️"

    return "Tidak Sarkasme ✅"

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# INPUT TEKS
# =========================
if menu == "Input Teks":
    text = st.text_area("Masukkan teks ulasan")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            emotion = predict_emotion(text)
            sarcasm = detect_sarcasm(text)

            st.success(f"🎯 Emosi: {emotion}")
            st.info(f"🧠 Sarkasme: {sarcasm}")

# =========================
# DATASET
# =========================
elif menu == "Upload Dataset":
    file = st.file_uploader("Upload CSV (kolom: text)")

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("Kolom 'text' tidak ada")
            st.stop()

        if st.button("Proses"):
            df["emosi"] = df["text"].astype(str).apply(predict_emotion)
            df["sarkasme"] = df["text"].astype(str).apply(detect_sarcasm)

            # clustering sederhana (encoding label)
            df["emosi_num"] = df["emosi"].astype("category").cat.codes

            kmeans = KMeans(n_clusters=3, random_state=42)
            df["cluster"] = kmeans.fit_predict(df[["emosi_num"]])

            st.dataframe(df.head())

            # visualisasi
            fig, ax = plt.subplots()
            df["cluster"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
