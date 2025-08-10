import streamlit as st
import pandas as pd
from typing import Optional, Set

REQUIRED_COLUMNS: Set[str] = {"Symbol", "Sector", "Industry Group", "Industry"}


@st.cache_data
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load Excel into a DataFrame."""
    if uploaded_file is None:
        return None
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        return None


def validate_columns(df: pd.DataFrame) -> bool:
    """Return True if all required columns are present."""
    return REQUIRED_COLUMNS.issubset(set(df.columns))
