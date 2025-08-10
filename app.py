import streamlit as st
from visualization import show_main_app

# App configuration
st.set_page_config(page_title="Financial Stock Screener", layout="wide")

def main():
    show_main_app()

if __name__ == "__main__":
    main()
