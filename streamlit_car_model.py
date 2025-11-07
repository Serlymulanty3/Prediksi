# streamlit_car_chat_only_autoload_data.py
"""
Chat Prediksi Mobil — Chat Only (Auto-load model + dataset-aware)
- Auto-load: model.pkl, columns.json, dan (opsional) toyota.csv dari folder yang sama.
- Chat dibatasi ke dataset & model saja (guard).
- Pertanyaan seperti "model apa saja?" atau "tahun berapa saja?" dijawab LANGSUNG dari dataset (tanpa LLM).
- Jika LLM mengeluarkan blok JSON {"features": {...}}, app otomatis memprediksi (jika model tersedia).
"""

from __future__ import annotations
import os, io, json, re
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ==== LLM (Gemini) ====
try:
    import google.generativeai as genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

APP_TITLE = "Chat Prediksi Mobil — Data-Aware (Chat Only, Auto Load)"

# Dataset columns (default / expected)
EXPECTED_COLUMNS = ["model","year","price","transmission","mileage","fuelType","tax","mpg","engineSize"]

# ===== Helpers - Caching loaders =====
@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    """Cache by path string (hashable)"""
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_columns_from_path(columns_path: str) -> Optional[List[str]]:
    try:
        with open(columns_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "columns" in data:
            return list(data["columns"])
        if isinstance(data, list):
            return list(data)
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def load_dataset_from_path(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None

def ensure_dataframe(features: Dict[str, Any], expected_cols: Optional[List[str]]) -> pd.DataFrame:
    if expected_cols:
        row = {c: features.get(c, np.nan) for c in expected_cols}
        return pd.DataFrame([row])
    return pd.DataFrame([features])

def run_predict(features: Dict[str, Any], model, expected_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    df = ensure_dataframe(features, expected_cols)
    try:
        y_pred = model.predict(df)
        y_val = float(y_pred[0]) if hasattr(y_pred, "__iter__") else float(y_pred)
        return {"ok": True, "prediction": y_val, "df": df}
    except Exception as e:
        return {"ok": False, "error": str(e), "df": df}

JSON_BLOCK_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
def try_extract_features_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    m = JSON_BLOCK_RE.search(text)
    if not m: return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, dict) and "features" in data and isinstance(data["features"], dict):
            return data["features"]
    except Exception:
        return None
    return None

# ===== On-topic guard =====
ALLOWED_KEYWORDS = {
    "dataset","data","fitur","feature","features","kolom","columns","model.pkl","model pkl","prediksi","predict","prediction",
    "mobil","harga","price","regresi","randomforest","random forest","pipeline","onehot","ohe","scaler","year","mileage","transmission","fueltype","engine","enginesize","tax","mpg","toyota",
    "model apa saja","pilihan model","list model","daftar model","tahun apa saja","range tahun","tahun tersedia","daftar tahun"
}
REFUSAL_TEXT = "Maaf, saya hanya melayani pertanyaan seputar dataset dan model prediksi mobil ini."
def is_on_topic(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ALLOWED_KEYWORDS)

# ===== Simple intent detector for dataset Q&A =====
MODEL_PAT = re.compile(r"\b(model|tipe)\b.*\b(apa|apa saja|apa aja|yang ada|pilihan|list|daftar)\b", re.IGNORECASE)
YEAR_PAT  = re.compile(r"\b(tahun)\b.*\b(apa|apa saja|apa aja|yang ada|pilihan|list|daftar|berapa|range)\b", re.IGNORECASE)
COLS_PAT  = re.compile(r"\b(kolom|fitur|columns?)\b.*\b(apa|apa saja|apa aja|yang ada|list|daftar)\b", re.IGNORECASE)

def dataset_models_years_cols_answer(user_text: str, df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = (user_text or "").lower()
    parts = []

    if MODEL_PAT.search(user_text) or ("model" in lower and ("apa" in lower or "pilihan" in lower or "daftar" in lower or "list" in lower)):
        if "model" in df.columns:
            top_models = (df["model"].astype(str).value_counts().head(20)).index.tolist()
            parts.append("**Model tersedia (Top 20 by frequency)**: " + ", ".join(top_models))
        else:
            parts.append("Kolom `model` tidak ditemukan di dataset.")

    if YEAR_PAT.search(user_text) or ("tahun" in lower and ("apa" in lower or "pilihan" in lower or "daftar" in lower or "range" in lower or "berapa" in lower)):
        if "year" in df.columns:
            years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique().tolist())
            if years:
                parts.append(f"**Tahun tersedia**: {years[0]} — {years[-1]} (unik: {len(years)} tahun)")
            else:
                parts.append("Tidak ada data tahun yang valid.")
        else:
            parts.append("Kolom `year` tidak ditemukan di dataset.")

    if COLS_PAT.search(user_text):
        parts.append("**Kolom dataset**: " + ", ".join(df.columns.astype(str).tolist()))

    if parts:
        return "\n\n".join(parts)
    return None

# ===== LLM client =====
SYSTEM_INSTRUCTION_TEMPLATE = """
Anda adalah asisten pakar valuasi mobil bekas untuk dataset ini.
Batasan ketat: HANYA bahas dataset (kolom: {columns}) dan prediksi model.pkl. 
Jika user bertanya di luar topik, jawab: "Maaf, saya hanya melayani pertanyaan seputar dataset dan model prediksi mobil ini."
Gunakan daftar model/tahun berikut sebagai acuan nyata dari dataset (jika ada):
- Model (Top): {models_top}
- Tahun: {year_min} — {year_max} (unik: {year_count})

Saat fitur cukup, keluarkan blok JSON 'features' agar aplikasi memprediksi.
Bahasa Indonesia, ringkas, bertahap.
""".strip()

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, system_instruction: str):
    if not HAVE_GENAI:
        return None
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name, system_instruction=system_instruction)
    except Exception:
        return None

# ===== UI (CHAT ONLY) =====
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
st.title(APP_TITLE)
st.caption("Auto-load model & dataset. Chat dibatasi hanya seputar dataset & model. Pertanyaan 'model/tahun apa saja' dijawab dari data.")

# Resolve paths (folder yang sama)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")
CSV_PATH = os.path.join(BASE_DIR, "toyota.csv")

# Sidebar status
with st.sidebar:
    st.subheader("📁 Status Berkas")
    st.write(f"Folder: `{BASE_DIR}`")
    st.write(f"model.pkl: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")
    st.write(f"columns.json: {'✅' if os.path.exists(COLUMNS_PATH) else '❌'}")
    st.write(f"toyota.csv: {'✅' if os.path.exists(CSV_PATH) else '❌'}")
    st.divider()
    st.subheader("🤖 Gemini")
    llm_enabled = st.toggle("Aktifkan Gemini Chat", value=True)
    model_name = st.selectbox("Model LLM", ["gemini-2.5-flash","gemini-flash-latest","gemini-2.5-pro","gemini-pro-latest"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.5, 0.1)
    if llm_enabled:
        if not HAVE_GENAI:
            st.warning("Paket google-generativeai belum terpasang. Jalankan: python -m pip install google-generativeai")
        elif not os.environ.get("GEMINI_API_KEY"):
            st.warning("GEMINI_API_KEY belum diset di environment.")

# Load resources
model = load_model_from_path(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
columns = load_columns_from_path(COLUMNS_PATH) if os.path.exists(COLUMNS_PATH) else None
df = load_dataset_from_path(CSV_PATH) if os.path.exists(CSV_PATH) else None

# Prepare dataset context for LLM system prompt
if df is not None and not df.empty:
    try:
        top_models = (df["model"].astype(str).value_counts().head(15)).index.tolist() if "model" in df.columns else []
        years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique().tolist()) if "year" in df.columns else []
        y_min, y_max, y_count = (years[0], years[-1], len(years)) if years else ("-", "-", 0)
    except Exception:
        top_models, y_min, y_max, y_count = [], "-", "-", 0
else:
    top_models, y_min, y_max, y_count = [], "-", "-", 0

system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
    columns=", ".join(list(df.columns) if df is not None else EXPECTED_COLUMNS),
    models_top=", ".join(top_models) if top_models else "(tidak ada)",
    year_min=y_min, year_max=y_max, year_count=y_count
)

# Init LLM
llm = get_llm(model_name, system_instruction) if llm_enabled else None

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_msg = st.chat_input("Tulis pesan Anda… (hanya seputar dataset & model)")
def is_on_topic(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ALLOWED_KEYWORDS) or any(k in t for k in (["tahun","model","fitur","kolom","prediksi","harga","mobil"]))

if user_msg:
    # Guard topik
    if not is_on_topic(user_msg):
        st.session_state.messages.append({"role":"user","content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(REFUSAL_TEXT)
        st.session_state.messages.append({"role":"assistant","content":REFUSAL_TEXT})
    else:
        st.session_state.messages.append({"role":"user","content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # 1) Coba jawab langsung dari dataset (model/tahun/kolom)
        direct = dataset_models_years_cols_answer(user_msg, df)
        if direct:
            with st.chat_message("assistant"):
                st.markdown(direct)
            st.session_state.messages.append({"role":"assistant","content":direct})
        else:
            # 2) Kalau tidak cocok intent spesifik, pakai LLM (dengan instruction + context)
            reply_text = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                if llm is None:
                    reply_text = ("LLM tidak aktif. Aktifkan Gemini & set GEMINI_API_KEY.")
                    placeholder.markdown(reply_text)
                else:
                    # Build history
                    hist = []
                    for msg in st.session_state.messages:
                        role = "user" if msg["role"]=="user" else "model"
                        part = msg["content"]
                        hist.append({"role": role, "parts":[part]})
                    try:
                        chat = llm.start_chat(history=hist)
                        resp = chat.send_message(user_msg, generation_config={"temperature": temperature})
                        reply_text = getattr(resp, "text", None) or "(Tidak ada teks dari model)"
                    except Exception as e:
                        reply_text = f"Terjadi error memanggil LLM: {e}"
                    placeholder.markdown(reply_text)

            st.session_state.messages.append({"role":"assistant","content":reply_text})

            # Auto-predict jika LLM keluarkan blok JSON
            feats = try_extract_features_from_text(reply_text)
            if feats and model is not None:
                with st.chat_message("assistant"):
                    st.markdown("Menjalankan prediksi berdasarkan fitur yang disarankan…")
                    res = run_predict(feats, model, columns)
                    if res["ok"]:
                        st.success(f"Perkiraan harga: {res['prediction']:.2f}")
                        st.dataframe(res["df"])
                    else:
                        st.error("Gagal prediksi: " + res["error"])
                        st.dataframe(res["df"])
            elif feats and model is None:
                with st.chat_message("assistant"):
                    st.warning("Fitur dari LLM ditemukan, tetapi file model.pkl belum tersedia di folder ini.")
