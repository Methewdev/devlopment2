import os
import streamlit as st
import requests
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi & Segmentasi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah (API Mode)")

# =========================
# HUGGINGFACE CONFIG
# =========================
HF_TOKEN = st.secrets.get("HF_TOKEN", "")  # simpan di Streamlit Secrets
API_URL = "https://api-inference.huggingface.co/models/envidevelopment/sentiment-banking"

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =========================
# PREDICT FUNCTION (API)
# =========================
@st.cache_data(show_spinner=False)
def predict_api(text):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=20)

        if response.status_code != 200:
            return {"error": f"API Error: {response.text}"}

        result = response.json()

        # Format hasil
        if isinstance(result, list):
            result = result[0]

        return {item["label"]: float(item["score"]) for item in result}

    except Exception as e:
        return {"error": str(e)}

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# MODE 1: INPUT TEKS
# =========================
if menu == "Input Teks":
    st.subheader("📝 Analisis Satu Ulasan")

    text = st.text_area("Masukkan teks ulasan")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            with st.spinner("Memproses..."):
                result = predict_api(text)

            if "error" in result:
                st.error(result["error"])
            else:
                st.write("### 🔥 Probabilitas Emosi")
                st.json(result)

                label = max(result, key=result.get)
                st.success(f"🎯 Emosi Dominan: {label}")

# =========================
# MODE 2: UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    st.subheader("📂 Upload Dataset (CSV)")

    file = st.file_uploader("Upload CSV (harus ada kolom: text)")

    if file:
        try:
            df = pd.read_csv(file)

            if "text" not in df.columns:
                st.error("CSV harus memiliki kolom 'text'")
                st.stop()

            st.write("Preview Data:")
            st.dataframe(df.head())

            if st.button("🚀 Proses Analisis"):
                results = []
                progress = st.progress(0)

                for i, txt in enumerate(df["text"].astype(str)):
                    res = predict_api(txt)
                    results.append(res)
                    progress.progress((i + 1) / len(df))

                emotion_df = pd.DataFrame(results)

                if "error" in emotion_df.columns:
                    st.error("Ada error pada API, cek token/model")
                    st.stop()

                df = pd.concat([df, emotion_df], axis=1)

                # =========================
                # CLUSTERING
                # =========================
                emotion_cols = list(emotion_df.columns)

                kmeans = KMeans(n_clusters=3, random_state=42)
                df["cluster"] = kmeans.fit_predict(df[emotion_cols])

                st.success("Analisis selesai!")

                # =========================
                # OUTPUT
                # =========================
                st.write("### 📊 Hasil")
                st.dataframe(df.head())

                # =========================
                # VISUALISASI
                # =========================
                st.write("### 📈 Distribusi Cluster")

                fig, ax = plt.subplots()
                df["cluster"].value_counts().plot(kind="bar", ax=ax)
                st.pyplot(fig)

                st.write("### 🧠 Karakteristik Cluster")
                summary = df.groupby("cluster")[emotion_cols].mean()
                st.dataframe(summary)

                # =========================
                # DOWNLOAD
                # =========================
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hasil",
                    csv,
                    "hasil_analisis.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")
