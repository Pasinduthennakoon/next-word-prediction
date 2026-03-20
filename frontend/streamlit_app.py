import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://localhost:8000/generate-text"

st.set_page_config(page_title="AI Text Generator", layout="centered")

st.title("Cricket Text Generator")

# User inputs
input_text = st.text_area("Enter your prompt:", height=150)
word_count = st.slider("Number of words to generate:", 1, 200, 10)

# Button
if st.button("Generate"):
    if not input_text.strip():
        st.warning("Please enter some text")
    else:
        payload = {
            "input_text": input_text,
            "word_count": word_count
        }

        try:
            with st.spinner("Generating text..."):
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Generated Text:")
                    st.write(result)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Connection error: {e}")