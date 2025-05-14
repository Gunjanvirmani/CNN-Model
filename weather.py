import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="🌦️ Weather Condition Detector", layout="centered")
st.title("🌤️ Weather Image Classifier")
st.markdown("Upload an atmospheric image to classify it as fog/smog, lightning, rain, sandstorm, or snow.")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("weather_classification_model.h5")  # <-- model name for weather
    return model

model = load_model()

# Use the correct class order from your training
classes = ['fogsmog', 'lightning', 'rain', 'sandstorm', 'snow']  # <-- your categories

# File upload UI
uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Debug: Show the uploaded file name
        st.write("File uploaded successfully!")
        st.write(f"File name: {uploaded_file.name}")
        
        # Open the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        IMG_SIZE = 256
        img = np.array(image)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Only if you used rescale=1./255 during training
        img = np.expand_dims(img, axis=0)

        # Make prediction
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)
        predicted_class = classes[predicted_index]
        confidence = float(np.max(predictions))

        # Display results
        st.markdown("---")
        st.subheader("☁️ Prediction Result")
        st.success(f"**Class:** `{predicted_class}`")
        st.info(f"**Confidence:** `{confidence:.2f}`")

        # Optional: Show confidence for all classes
        st.markdown("### 📊 Confidence Scores")
        for i, label in enumerate(classes):
            st.write(f"{label}: `{predictions[0][i]:.2f}`")

    except Exception as e:
        st.error(f"Error loading or processing image: {e}")
else:
    st.info("Please upload an atmospheric image to classify.")
