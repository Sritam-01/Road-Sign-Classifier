import streamlit as st
import requests

st.title("Road Sign Classification System")

uploaded_file = st.file_uploader(
    "Upload Road Sign Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    st.image(uploaded_file,
             caption="Uploaded Image",
             use_container_width=True)

    files = {
        'file': uploaded_file.getvalue()
    }

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        files=files
    )

    result = response.json()

    st.success(
        f"Prediction: {result['prediction']}"
    )