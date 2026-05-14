import streamlit as st
# import requests
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Road Sign Classification System")

# LOAD MODEL
model = tf.keras.models.load_model(
    "model/road_sign_model.h5"
)
# CLASS LABELS
classes = {
    0:'Speed limit 20 km/h',
    1:'Speed limit 30 km/h',
    2:'Speed limit 50 km/h',
    3:'Speed limit 60 km/h',
    4:'Speed limit 70 km/h',
    5:'Speed limit 80 km/h',
    6:'End of speed limit 80 km/h',
    7:'Speed limit 100 km/h',
    8:'Speed limit 120 km/h',
    9:'No passing',
    10:'No passing for vehicles over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Vehicles over 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End of all speed and passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End of no passing by vehicles over 3.5 tons'
}

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Road Sign Image",
    type=["jpg", "png", "jpeg"]
)

# RUN ONLY AFTER IMAGE UPLOAD
if uploaded_file is not None:

    st.image(uploaded_file,
             caption="Uploaded Image",
             use_container_width=True)

    # files = {
    #     'file': uploaded_file.getvalue()
    # }
    # PROCESS IMAGE
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((30,30))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    # response = requests.post(
    #     "http://127.0.0.1:5000/predict",
    #     files=files
    # )
    # PREDICTION
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # result = response.json()
    # GET LABEL
    label = classes.get(
        predicted_class,
        f"Class {predicted_class}"
    )

    # st.success(
    #     f"Prediction: {result['prediction']}"
    # )
    # SHOW RESULT
    st.success(f"Prediction: {label}")