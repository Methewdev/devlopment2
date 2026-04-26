import streamlit as st
import pandas as pd
import requests
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi & Segmentasi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah")

# =========================
# API MODEL (STABIL)
# =========================
API_URL = "https://api-inference.huggingface.co/models/w11wo/indonesian-roberta-base-sentiment-classifier"

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# =========================
# SAFE API CALL
# =========================
def call_api(text):
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text},
            timeout=20
        )

        # cek status
        if response.status_code != 200:
            return {"error": f"{response.status_code}: {response.text}"}

        # cek kosong
        if not response.text.strip():
            return {"error": "Response kosong"}

        try:
            result = response.json()
        except:
            return {"error": "Response bukan JSON"}

        return result

    except Exception as e:
        return {"error": str(e)}

# =========================
# PREDICT FUNCTION
# =========================
def predict(text):
    result = call_api(text)

    if isinstance(result, dict) and "error" in result:
        return result

    try:
        if isinstance(result, list):
            result = result[0]

        return {
            item.get("label", f"class_{i}"): float(item.get("score", 0))
            for i, item in enumerate(result)
        }

    except:
        return {"error": "Format output tidak dikenali"}

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# INPUT TEKS
# =========================
if menu == "Input Teks":
    st.subheader("📝 Analisis Satu Ulasan")

    text = st.text_area("Masukkan teks")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            with st.spinner("Memproses..."):
                result = predict(text)

            if "error" in result:
                st.error(result["error"])
            else:
                st.write("### 🔥 Probabilitas Emosi")
                st.json(result)

                label = max(result, key=result.get)
                st.success(f"🎯 Emosi Dominan: {label}")

# =========================
# UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    st.subheader("📂 Upload Dataset CSV")

    file = st.file_uploader("Upload CSV (kolom wajib: text)")

    if file:
        try:
            df = pd.read_csv(file)

            if "text" not in df.columns:
                st.error("CSV harus memiliki kolom 'text'")
                st.stop()

            st.write("### Preview Data")
            st.dataframe(df.head())

            if st.button("🚀 Proses Analisis"):
                results = []
                progress = st.progress(0)

                for i, txt in enumerate(df["text"].astype(str)):
                    res = predict(txt)
                    results.append(res)
                    progress.progress((i + 1) / len(df))

                emotion_df = pd.DataFrame(results)

                # cek error kolom
                if "error" in emotion_df.columns:
                    st.error("Sebagian data gagal diproses")
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
                st.write("### 📊 Hasil Data")
                st.dataframe(df.head())

                # =========================
                # VISUALISASI
                # =========================
                st.write("### 📈 Distribusi Cluster")
                fig, ax = plt.subplots()
                df["cluster"].value_counts().plot(kind="bar", ax=ax)
                st.pyplot(fig)

                st.write("### 🧠 Karakteristik Emosi per Cluster")
                summary = df.groupby("cluster")[emotion_cols].mean()
                st.dataframe(summary)

                # =========================
                # DOWNLOAD
                # =========================
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Hasil", csv, "hasil_analisis.csv")

        except Exception as e:
            st.error(f"Error membaca file: {e}")
