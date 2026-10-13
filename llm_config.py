
import os
import streamlit as st
from crewai import LLM


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    key = key or os.getenv("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to "
            "Streamlit Cloud Secrets."
        )

    return key


def build_llm():
    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=get_api_key(),
        temperature=0.1,
    )

