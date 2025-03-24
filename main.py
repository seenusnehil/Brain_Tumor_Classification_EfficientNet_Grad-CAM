import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('brain_tumor_model.keras')

# Streamlit UI for prediction
st.title("Brain Tumor Classification")

# File upload to input an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image and make predictions
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
