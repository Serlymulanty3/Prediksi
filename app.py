import streamlit as st
import pandas as pd
import requests
import json
import joblib
import pickle
import io
import os
import tempfile
from typing import List, Dict, Any

# =========================================
# HUGGINGFACE MODEL RAW LINK
# =========================================
HF_MODEL_URL = "https://huggingface.co/Ranzzz/prediksi/resolve/main/model.pkl"


# =========================================================
# ✅ Fungsi download model dari HuggingFace (versi stabil)
# =========================================================
def download_model_hf(hf_url: str) -> str:
    try:
        # Tambahkan ?download=1 agar HuggingFace mengembalikan raw file
        if not hf_url.endswith("?download=1"):
            download_url = hf_url + "?download=1"
        else:
            download_url = hf_url

        # Ambil nama file
        filename = hf_url.split("/")[-1]
        temp_path = os.path.join(tempfile.gettempdir(), filename)

        # Jika sudah ada → skip download
        if os.path.exists(temp_path):
            return temp_path

        # Download
        response = requests.get(download_url, stream=True)

        if response.status_code != 200:
            st.error(f"Gagal download: status {response.status_code}")
            return None

        # Simpan ke file sementara
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

        return temp_path

    except Exception as e:
        st.error(f"Error download model HuggingFace: {e}")
        return None


# =========================================================
# ✅ Fungsi load model (fallback joblib → pickle)
# =========================================================
@st.cache_resource
def load_model(path: str, hf_url: str = None):
    # Coba load dari lokal
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Gagal load model lokal: {e}")

    # Jika gagal → download dari HF
    if hf_url:
        st.info("Mengunduh model dari HuggingFace...")
        downloaded_path = download_model_hf(hf_url)

        if downloaded_path:
            try:
                return joblib.load(downloaded_path)
            except Exception:
                try:
                    with open(downloaded_path, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    st.error(f"Gagal load model dari HuggingFace: {e}")
                    return None

    st.error("Model tidak ditemukan.")
    return None


# =================================================
