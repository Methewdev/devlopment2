import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
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

    if ("bagus" in text or "mantap" in text) and ("error" in text or "lambat" in text):
        return "Sarkasme Terdeteksi ⚠️"

    return "Tidak Sarkasme ✅"

# =========================
# WARNA OUTPUT
# =========================
def show_emotion(emotion):
    if "Senang" in emotion:
        st.success(f"🟢 Emosi: {emotion}")
    elif "Marah" in emotion:
        st.error(f"🔴 Emosi: {emotion}")
    elif "Kecewa" in emotion:
        st.warning(f"🟠 Emosi: {emotion}")
    else:
        st.info(f"🔵 Emosi: {emotion}")

def show_sarcasm(sarcasm):
    if "Sarkasme" in sarcasm:
        st.markdown(f"### 🟣 {sarcasm}")
    else:
        st.markdown(f"### 🟢 {sarcasm}")

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# INPUT TEKS
# =========================
if menu == "Input Teks":
    st.subheader("📝 Analisis Satu Ulasan")

    text = st.text_area("Masukkan teks ulasan")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            emotion = predict_emotion(text)
            sarcasm = detect_sarcasm(text)

            show_emotion(emotion)
            show_sarcasm(sarcasm)

# =========================
# UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    st.subheader("📂 Upload Dataset CSV")

    file = st.file_uploader("Upload CSV (kolom wajib: text)")

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV harus memiliki kolom 'text'")
            st.stop()

        st.write("### Preview Data")
        st.dataframe(df.head())

        if st.button("🚀 Proses Analisis"):
            df["emosi"] = df["text"].astype(str).apply(predict_emotion)
            df["sarkasme"] = df["text"].astype(str).apply(detect_sarcasm)

            # encoding untuk clustering
            df["emosi_num"] = df["emosi"].astype("category").cat.codes

            # =========================
            # CLUSTERING
            # =========================
            kmeans = KMeans(n_clusters=3, random_state=42)
            df["cluster"] = kmeans.fit_predict(df[["emosi_num"]])

            st.success("Analisis selesai!")

            # =========================
            # OUTPUT
            # =========================
            st.write("### 📊 Hasil Data")
            st.dataframe(df.head())

            # =========================
            # VISUALISASI
            # =========================
            st.write("### 📈 Distribusi Cluster")
            fig, ax = plt.subplots()
            df["cluster"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

            st.write("### 🧠 Ringkasan Emosi per Cluster")
            summary = df.groupby("cluster")["emosi"].value_counts().unstack().fillna(0)
            st.dataframe(summary)

            # =========================
            # DOWNLOAD
            # =========================
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil", csv, "hasil_analisis.csv")
