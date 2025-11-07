import os
import json
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
import requests
import tempfile

# =========================================
# HUGGINGFACE MODEL URL
# =========================================
HF_MODEL_URL = "https://huggingface.co/Ranzzz/prediksi/resolve/main/model.pkl"

# Set GEMINI KEY contoh
os.environ["GEMINI_API_KEY"] = "AIzaSyA50jHU6-rZzngv8wejAIsyX75GYRsYJY4"

# =========================================
# FUNGSI CHAT AI (OpenAI & Gemini)
# =========================================
def chat_reply(system_prompt: str, messages: list, api_key: str, model="gpt-4o-mini", provider="openai"):
    if provider == "openai":
        import openai
        openai.api_key = api_key
        conversation = [{"role": "system", "content": system_prompt}] + messages
        resp = openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            max_tokens=600,
            temperature=0.2,
        )
        return resp.choices[0].message["content"].strip()

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        prompt_text = system_prompt + "\n\n"
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt_text += f"{role.upper()}: {content}\n"
        model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model_gemini.generate_content(prompt_text)
        return response.text.strip()

    else:
        raise ValueError("Provider tidak dikenali. Gunakan 'openai' atau 'gemini'.")

# =========================================
# FUNGSI DOWNLOAD MODEL DARI HUGGINGFACE
# =========================================
def download_model_from_hf(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()

        temp_path = os.path.join(tempfile.gettempdir(), "hf_model.pkl")
        with open(temp_path, "wb") as f:
            f.write(response.content)

        return temp_path
    except Exception as e:
        st.error(f"Gagal download model dari HuggingFace: {e}")
        return None

# =========================================
# FUNGSI LOAD MODEL (LOKAL + HF)
# =========================================
@st.cache_resource
def load_model(path: str, hf_url: str = None):
    # Jika file lokal tidak ada → download dari HuggingFace
    if not os.path.exists(path):
        if hf_url:
            st.warning("Model lokal tidak ditemukan. Mengunduh dari HuggingFace...")
            downloaded_path = download_model_from_hf(hf_url)
            if downloaded_path:
                return joblib.load(downloaded_path)
            else:
                st.error("Gagal mengunduh model dari HuggingFace.")
                return None
        else:
            st.error("Model lokal tidak ditemukan dan HF URL tidak ada.")
            return None

    return joblib.load(path)

# =========================================
# KONFIG STREAMLIT
# =========================================
st.set_page_config(page_title="Prediksi Harga Mobil + Chat AI", layout="wide")
st.title("🚗 Prediksi Harga Mobil & Chat AI")

# Sidebar
st.sidebar.header("⚙ Konfigurasi")
model_path = st.sidebar.text_input("Path ke model (.pkl)", value="model.pkl")
example_json_path = st.sidebar.text_input("Path example_features.json", value="example_features.json")

openai_key_env = st.sidebar.text_input("Masukkan API Key (opsional)", value="", type="password")
ai_provider = st.sidebar.selectbox("Pilih provider AI", ["openai", "gemini"], index=0)

# Gunakan ENV jika kosong
if ai_provider == "gemini":
    use_env_key = st.sidebar.checkbox("Gunakan dari ENV var GEMINI_API_KEY jika kosong", value=True)
else:
    use_env_key = st.sidebar.checkbox("Gunakan dari ENV var OPENAI_API_KEY jika kosong", value=True)

if use_env_key and not openai_key_env:
    if ai_provider == "gemini":
        openai_key_env = os.environ.get("GEMINI_API_KEY", "")
    else:
        openai_key_env = os.environ.get("OPENAI_API_KEY", "")

# =========================================
# LOAD SCHEMA
# =========================================
example_schema = {}
try:
    with open(example_json_path, "r") as f:
        example_schema = json.load(f)
except:
    example_schema = {
        "model": "Aygo",
        "year": 2017,
        "transmission": "Manual",
        "mileage": 11730,
        "fuelType": "Petrol",
        "tax": 0,
        "mpg": 68.9,
        "engineSize": 1.0
    }
    st.sidebar.warning("Gagal load example_features.json. Menggunakan default.")

# =========================================
# LOAD MODEL
# =========================================
model = load_model(model_path, hf_url=HF_MODEL_URL)
if model is None:
    st.error("❌ Model gagal dimuat dari lokal maupun HuggingFace.")

# =========================================
# LAYOUT UTAMA
# =========================================
left, right = st.columns([1.2, 1.0])

# =========================================
# KIRI: INPUT FITUR
# =========================================
with left:
    st.subheader("📊 Input Fitur Mobil")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model", value=example_schema.get("model"))
            year = st.number_input("Tahun", min_value=1990, max_value=2025, value=int(example_schema.get("year")))
            transmission = st.selectbox("Transmisi", ["Manual", "Automatic", "Semi-Auto"])
            fuel_type = st.selectbox("Jenis Bahan Bakar", ["Petrol", "Diesel", "Hybrid", "Electric"])
        with col2:
            mileage = st.number_input("Mileage", min_value=0, value=int(example_schema.get("mileage")))
            tax = st.number_input("Pajak (£)", min_value=0, value=int(example_schema.get("tax")))
            mpg = st.number_input("MPG", min_value=0.0, value=float(example_schema.get("mpg")))
            engine_size = st.number_input("Engine Size", min_value=0.0, value=float(example_schema.get("engineSize")))

        submitted = st.form_submit_button("🚗 Prediksi Harga Mobil")

    if submitted:
        input_dict = {
            "model": model_name,
            "year": year,
            "transmission": transmission,
            "mileage": mileage,
            "fuelType": fuel_type,
            "tax": tax,
            "mpg": mpg,
            "engineSize": engine_size
        }

        X_df = pd.DataFrame([input_dict])

        try:
            pred = model.predict(X_df)
            price = float(pred[0])
            st.success(f"💰 Prediksi harga mobil: £{price:,.2f}")
            st.session_state["last_prediction"] = {"price": price, "input": input_dict}
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

        if openai_key_env and st.button("🤖 Analisis Otomatis dengan AI"):
            system_prompt = (
                f"Jelaskan hasil prediksi harga mobil.\n"
                f"Hasil prediksi: £{price:,.2f}.\n"
                f"Fitur: {json.dumps(input_dict, ensure_ascii=False)}."
            )
            try:
                ai_reply = chat_reply(system_prompt, [], openai_key_env, provider=ai_provider)
                st.info(ai_reply)
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")

# =========================================
# KANAN: CHAT AI
# =========================================
with right:
    st.subheader("Chat dengan AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**Kamu:** {text}")
        else:
            st.markdown(f"**AI:** {text}")

    user_input = st.text_area("Ketik pesan", height=120)
    col1, col2 = st.columns(2)

    with col1:
        send_btn = st.button("Kirim ke AI")
    with col2:
        clear_btn = st.button("Hapus chat")

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    if send_btn and user_input.strip():
        st.session_state.chat_history.append(("user", user_input))

        system_prompt = (
            "Kamu adalah AI yang menjelaskan prediksi harga mobil secara sederhana."
        )

        last_pred = st.session_state.get("last_prediction")
        if last_pred:
            system_prompt += (
                f"\nPrediksi terakhir: £{last_pred['price']:,.2f}, fitur: "
                f"{json.dumps(last_pred['input'], ensure_ascii=False)}"
            )

        messages = [{"role": r, "content": t} for r, t in st.session_state.chat_history[-10:]]

        if not openai_key_env:
            st.error("API Key belum diisi!")
        else:
            try:
                ai_reply = chat_reply(system_prompt, messages, openai_key_env, provider=ai_provider)
                st.session_state.chat_history.append(("assistant", ai_reply))
                st.rerun()
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown(
    """
    *Panduan:*
    - Simpan model lokal jika ingin load lokal.
    - Jika tidak ada → otomatis download dari HuggingFace.
    - Tidak perlu API Key HuggingFace jika repo publik.
    - Jalankan aplikasi dengan:

      ```
      streamlit run app.py
      ```
    """
)
