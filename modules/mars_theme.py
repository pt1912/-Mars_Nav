from pathlib import Path

import streamlit as st


# =====================================================
# ARCADIA ICE EXPLORER
# Theme Manager
# =====================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSS_PATH = PROJECT_ROOT / "assets" / "mars_theme.css"


def apply_theme():

    """
    Load and apply the ARCADIA scientific theme.

    The base theme is configured in:
        .streamlit/config.toml

    Custom visual styles are loaded from:
        assets/mars_theme.css
    """

    if not CSS_PATH.is_file():

        raise FileNotFoundError(
            f"Theme stylesheet not found: {CSS_PATH}"
        )

    st.html(CSS_PATH)
