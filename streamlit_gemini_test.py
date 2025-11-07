# streamlit_gemini_test_v2.py
import os
import streamlit as st

st.set_page_config(page_title="Gemini Quick Test", page_icon="✨", layout="centered")
st.title("✨ Gemini Quick Test (Streamlit)")
st.caption("Aplikasi minimal untuk memastikan koneksi ke Gemini bekerja.")

# --- Import SDK ---
try:
    import google.generativeai as genai
except Exception:
    st.error("Paket `google-generativeai` belum terpasang. Jalankan:  `python -m pip install google-generativeai`")
    st.stop()

# --- Sidebar settings (simple) ---
with st.sidebar:
    st.subheader("⚙️ Pengaturan")
    api_present = bool(os.environ.get("GEMINI_API_KEY", ""))
    st.write(f"GEMINI_API_KEY terdeteksi: **{'YA' if api_present else 'TIDAK'}**")
    model_name = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"],
        index=0
    )
    temperature = st.slider("Level Inisiatif", 0.0, 2.0, 0.7, 0.1)

if not api_present:
    st.warning("Set environment variable `GEMINI_API_KEY` di terminal, lalu jalankan ulang aplikasi.")
    st.stop()

# --- Configure client ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)
except Exception as e:
    st.error(f"Gagal inisialisasi Gemini: {e}")
    st.stop()

# --- Simple Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content}

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ketik pesan untuk Gemini…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Call Gemini
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            resp = model.generate_content(
                user_text,
                generation_config={"temperature": temperature}
            )
            reply = getattr(resp, "text", "") or "(Tidak ada teks dari model)"
        except Exception as e:
            reply = f"❌ Error memanggil Gemini: {e}"
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.info("✅")
