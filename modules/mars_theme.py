"""Styling for the Mars-themed Streamlit application."""

from pathlib import Path
import streamlit as st


def apply_theme():
    css = Path(__file__).resolve().parents[1] / 'assets' / 'mars_theme.css'
    st.markdown(f'<style>{css.read_text(encoding="utf-8")}</style>',
                unsafe_allow_html=True)
