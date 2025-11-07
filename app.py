import streamlit as st
import pandas as pd
import requests
import json
import joblib
import os
import sklearn 
import tempfile
from typing import List, Dict, Any

# ✅ FIX UNTUK MODEL LOAD ERROR: Memastikan modul ini dimuat sebelum joblib.load
import sklearn.compose._column_transformer 

# =========================================
# HUGGINGFACE FILE LINKS
# =========================================
HF_MODEL_URL = "https://huggingface.co/Ranzzz/prediksi/resolve/main/model.pkl"

# FIX: kedua file ini TIDAK ADA, jadi harus fallback
HF_COLUMNS_URL = "https://huggingface.co/Ranzzz/prediksi/resolve/main/columns.json"
HF_EXAMPLE_URL = "https://huggingface.co/Ranzzz/prediksi/resolve/main/example_features.json"


# =========================================
# DOWNLOAD FILE
# =========================================
def download_file_with_progress(url: str, dest_path: str, chunk_size: int = 32768, timeout: int = 30):
    with st.spinner("📥 Mengunduh model dari HuggingFace..."):
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        progress_bar = st.progress(0)

        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress_bar.progress(min(int(downloaded / total * 100), 100))

        progress_bar.empty()
    return True


# =========================================
# MODEL LOADER (cached)
# =========================================
@st.cache_resource(show_spinner=False)
def get_local_model_path(hf_url: str) -> str:
    tmp_dir = tempfile.gettempdir()
    filename = "model.pkl"
    local_path = os.path.join(tmp_dir, filename)

    # If already downloaded and valid
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path

    # Otherwise download
    download_file_with_progress(hf_url, local_path)
    return local_path


@st.cache_resource(show_spinner=False)
def load_model_from_hf(hf_url: str):
    try:
        local_path = get_local_model_path(hf_url)
        return joblib.load(local_path)
    except Exception as e:
        raise RuntimeError(f"Load model gagal: {e}")


# =========================================
# JSON LOADER (with FIXED fallback)
# =========================================
@st.cache_data
def load_json_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None  # no warning spam


# =========================================
# FIX: fallback untuk columns.json & example.json
# =========================================
columns_json = load_json_url(HF_COLUMNS_URL)
example_json = load_json_url(HF_EXAMPLE_URL)

if not columns_json:
    # fallback: biarkan pipeline pakai raw dataframe
    columns_json = None

if not example_json:
    example_json = {
        "model": "Avanza",
        "year": 2018,
        "transmission": "Automatic",
        "mileage": 50000,
        "fuelType": "Petrol",
        "tax": 0,
        "mpg": 15.0,
        "engineSize": 1.5,
    }


# =========================================
# INPUT BUILDER (safe)
# =========================================
def build_model_input(raw_row: Dict[str, Any], model_columns: List[str]) -> pd.DataFrame:
    import numpy as np

    row = pd.Series(0, index=model_columns, dtype=float)

    for k, v in raw_row.items():
        if k in model_columns:
            try:
                row[k] = float(v)
            except:
                row[k] = 1.0

    for col in model_columns:
        if "_" in col:
            prefix, suffix = col.split("_", 1)
            if prefix in raw_row:
                raw_val = str(raw_row[prefix]).lower()
                if raw_val == suffix.lower():
                    row[col] = 1.0

    return pd.DataFrame([row])


# =========================================
# CHAT FUNCTION (REVISED FOR AUTOMATIC API KEY & FALLBACK)
# =========================================
def chat_reply(system_prompt, messages, api_key, provider="openai"):
    # --- LOGIKA FALLBACK OTOMATIS JIKA API KEY KOSONG ---
    if not api_key:
        last_pred_price = "belum ada"
        car_features = "tidak diketahui"
        
        # Cek langsung st.session_state untuk data prediksi terakhir
        if "last_pred" in st.session_state:
            lp = st.session_state["last_pred"]
            last_pred_price = f"Rp {lp['price']:,.0f}"
            car_features = ", ".join([f"{k}: {v}" for k, v in lp['input'].items() if k != 'model'])
            car_features = f"Model {lp['input']['model']}, {car_features}"
            
        
        # Respons otomatis/mock
        fallback_response = f"🤖 **(Mode Otomatis/Offline)**: Halo! API Key tidak ditemukan di environment.\n\n"
        fallback_response += f"Prediksi harga mobil Anda adalah **{last_pred_price}**.\n\n"
        fallback_response += "Saya hanya dapat memberikan respons generik ini. Untuk analisis AI yang sesungguhnya, mohon setel environment variable **`{}_API_KEY`**." .format(provider.upper())
        return fallback_response
    # ----------------------------------------------------

    # --- PANGGIL API LLM JIKA API KEY ADA ---
    if provider == "openai":
        import openai
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.2
        )
        return resp.choices[0].message["content"]

    else:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        user_message_content = messages[0]["content"] if messages else ""
        prompt = f"{system_prompt}\n\nUser: {user_message_content}"
        
        return genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt).text


# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="Prediksi Harga Mobil – Serly", layout="wide")
st.title("🚗 Prediksi Harga Mobil – Serly (Model HuggingFace)")

# ------------- SIDEBAR & API KEY SETUP -------------
st.sidebar.header("⚙ Pengaturan")
provider = st.sidebar.selectbox("Provider AI", ["openai", "gemini"])

# ✅ PERUBAHAN: Dapatkan API Key secara OTOMATIS dari Environment Variable
api_key = os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY", "")

# Tampilkan status koneksi di sidebar
if api_key:
    st.sidebar.success(f"✅ API Key ({provider.upper()}) terdeteksi.")
else:
    st.sidebar.warning(f"⚠️ API Key ({provider.upper()}) tidak ditemukan. Chat AI berjalan dalam Mode Otomatis.")

# ------------- LOAD MODEL -------------
try:
    model = load_model_from_hf(HF_MODEL_URL)
except Exception as e:
    st.error(f"❌ Model gagal dimuat: {e}")
    st.stop()


# =========================================
# INPUT FORM
# =========================================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📋 Input Fitur Mobil")

    with st.form("form_input"):
        model_name = st.text_input("Model", example_json["model"])
        year = st.number_input("Tahun", 1990, 2025, example_json["year"])
        transmission = st.selectbox("Transmisi", ["Manual", "Automatic"], index=0)
        mileage = st.number_input("Jarak Tempuh (km)", 0, 500000, example_json["mileage"])
        fuel = st.selectbox("Bahan Bakar", ["Petrol", "Diesel"], index=0)
        tax = st.number_input("Pajak", 0, 100000, example_json["tax"])
        mpg = st.number_input("MPG", 0.0, 999.0, example_json["mpg"])
        engine = st.number_input("Engine Size", 0.1, 10.0, example_json["engineSize"])

        submit = st.form_submit_button("🚗 Prediksi Harga")

    if submit:
        raw = {
            "model": model_name,
            "year": year,
            "transmission": transmission,
            "mileage": mileage,
            "fuelType": fuel,
            "tax": tax,
            "mpg": mpg,
            "engineSize": engine,
        }

        # FIX: jika columns.json tidak ada → pakai raw DataFrame
        if columns_json:
            X = build_model_input(raw, columns_json)
        else:
            X = pd.DataFrame([raw])

        try:
            pred = model.predict(X)[0]
            st.success(f"💰 Prediksi Harga Mobil: **Rp {pred:,.0f}**")
            # Simpan hasil prediksi dan input ke session state
            st.session_state["last_pred"] = {"price": pred, "input": raw}
        except Exception as e:
            st.error(f"❌ Error prediksi: {e}")


# =========================================
# CHAT AREA
# =========================================
with right:
    st.subheader("💬 Chat AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Tampilkan Riwayat Chat
    for role, text in st.session_state.chat_history:
        st.markdown(f"**{role.upper()}:** {text}")

    user_text = st.text_area("Tulis pesan...")

    if st.button("Kirim"):
        if not user_text:
            st.warning("Mohon tulis pesan Anda terlebih dahulu.")
        else:
            st.session_state.chat_history.append(("user", user_text))

            system_prompt = "Kamu adalah AI yang menjelaskan prediksi mobil."
            
            # Tambahkan konteks prediksi ke system_prompt HANYA jika ada prediksi terakhir (untuk LLM asli)
            if "last_pred" in st.session_state and api_key:
                lp = st.session_state["last_pred"]
                system_prompt += f"\nPrediksi terakhir: Rp {lp['price']:,.0f}. Fitur: {lp['input']}" 

            with st.spinner("🤖 AI sedang berpikir..."):
                # chat_reply akan otomatis memilih mode (API atau Fallback)
                reply = chat_reply(system_prompt, [{"role": "user", "content": user_text}], api_key, provider)
            
            st.session_state.chat_history.append(("ai", reply))
            
            # ✅ FIX: Ganti st.experimental_rerun() dengan st.rerun()
            st.rerun()


# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Aplikasi Prediksi Harga Mobil – Serly ✅ Stabil dan Fungsional")