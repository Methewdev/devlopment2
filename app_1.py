import os
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi & Segmentasi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah (Transformer)")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# LOAD MODEL (SAFE MODE)
# =========================
@st.cache_resource
def load_model():
    PRIMARY_MODEL = "envidevelopment/sentiment-banking"  # model kamu
    FALLBACK_MODEL = "indobenchmark/indobert-base-p1"   # model publik

    try:
        st.info("🔄 Loading model utama...")
        tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(PRIMARY_MODEL)
        st.success("✅ Model utama berhasil dimuat")

    except Exception as e:
        st.warning("⚠️ Model utama gagal, pakai fallback model publik")
        st.warning(str(e))

        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL,
            num_labels=6
        )

    model.to("cpu")
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# Label handling
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
# PREDICT FUNCTION
# =========================
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

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
    st.subheader("📝 Analisis Satu Ulasan")

    text = st.text_area("Masukkan teks ulasan")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("⚠️ Masukkan teks terlebih dahulu")
        else:
            result = predict(text)
            st.json(result)

            label = max(result, key=result.get)
            st.success(f"🎯 Emosi Dominan: {label}")

# =========================
# UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    st.subheader("📂 Upload Dataset CSV")

    file = st.file_uploader("Upload file CSV (kolom: text)")

    if file:
        try:
            df = pd.read_csv(file)

            if 'text' not in df.columns:
                st.error("❌ CSV harus punya kolom 'text'")
                st.stop()

            st.write("Preview:")
            st.dataframe(df.head())

            if st.button("🚀 Proses"):
                with st.spinner("Processing..."):

                    # Predict
                    results = df['text'].astype(str).apply(predict)
                    emotion_df = pd.DataFrame(list(results))

                    df = pd.concat([df, emotion_df], axis=1)

                    # Clustering
                    emotion_cols = list(emotion_df.columns)
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    df['cluster'] = kmeans.fit_predict(df[emotion_cols])

                st.success("✅ Selesai!")

                # Show data
                st.dataframe(df.head())

                # Plot cluster
                st.subheader("Distribusi Cluster")
                fig, ax = plt.subplots()
                df['cluster'].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)

                # Cluster summary
                st.subheader("Karakteristik Cluster")
                summary = df.groupby('cluster')[emotion_cols].mean()
                st.dataframe(summary)

                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil",
                    csv,
                    "hasil_analisis.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error("❌ Error membaca file")
            st.error(e)