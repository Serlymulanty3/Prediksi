import streamlit as st

st.set_page_config(page_title="Cek Streamlit", page_icon="✅", layout="centered")
st.title("✅ Streamlit jalan!")
st.write("Kalau kamu melihat halaman ini, berarti instalasi & port berjalan normal.")
if st.button("Tes tombol"):
    st.success("Tombol bekerja!")

st.info("Selanjutnya kita bisa jalankan app prediksi mobil.")