import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Analisis Emosi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah")

# =========================
# CONFIG API
# =========================
API_URL = "https://api-inference.huggingface.co/models/envidevelopment/sentiment-banking"
HEADERS = {"Authorization": "Bearer YOUR_HF_TOKEN"}  # isi token kamu

# =========================
# PREDICT VIA API
# =========================
def predict(text):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})

    if response.status_code != 200:
        return {"error": response.text}

    result = response.json()[0]

    return {item['label']: item['score'] for item in result}

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

            if "error" in result:
                st.error(result["error"])
            else:
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
            st.error("Kolom 'text' tidak ada")
            st.stop()

        if st.button("Proses"):
            results = df["text"].apply(predict)
            df["hasil"] = results

            st.dataframe(df.head())
