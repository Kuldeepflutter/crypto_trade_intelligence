import streamlit as st

st.set_page_config(
    page_title="Crypto Trade Intelligence",
    page_icon="🐋",
    layout="wide"
)

st.title("🐋 Crypto Trade Intelligence")
st.caption("Whale Intelligence & Market Scanner")

if st.button("Run Market Scan"):
    with st.spinner("Scanning markets..."):
        st.info("Connect your existing scan engine here.")

st.success("Dashboard interface is running.")
