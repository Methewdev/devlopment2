import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi & Segmentasi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah")

# =========================
# SIMPLE EMOTION RULE (STABLE)
# =========================
def predict(text):
    text = text.lower()

    emotions = {
        "senang": ["bagus", "mantap", "puas", "baik", "cepat"],
        "marah": ["buruk", "jelek", "parah", "error", "gagal"],
        "kecewa": ["kecewa", "lambat", "tidak", "kurang"],
        "netral": []
    }

    scores = {k: 0 for k in emotions}

    for emo, keywords in emotions.items():
        for word in keywords:
            if word in text:
                scores[emo] += 1

    # normalisasi
    total = sum(scores.values())
    if total == 0:
        scores["netral"] = 1
        total = 1

    return {k: v/total for k, v in scores.items()}

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
        result = predict(text)

        st.json(result)

        label = max(result, key=result.get)
        st.success(f"Emosi Dominan: {label}")

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
            results = df["text"].astype(str).apply(predict)
            emotion_df = pd.DataFrame(list(results))

            df = pd.concat([df, emotion_df], axis=1)

            # clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            df["cluster"] = kmeans.fit_predict(emotion_df)

            st.dataframe(df.head())

            # visualisasi
            fig, ax = plt.subplots()
            df["cluster"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
