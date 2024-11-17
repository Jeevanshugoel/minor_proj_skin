import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Apply a custom theme for the app
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        h1, h3, p {
            color: #FF0000; /* Red theme */
            text-align: center;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        footer:after {
            content: 'This app is in its early stage. We recommend consulting a professional for accurate diagnosis.';
            visibility: visible;
            display: block;
            position: relative;
            color: white;
            background-color: #FF0000;
            text-align: center;
            padding: 10px;
            margin-top: 10px;
        }
        .stButton button {
            background-color: #FF0000;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #A00000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Download and load the pre-trained model dynamically
@st.cache_resource
def load_model():
    model_path = "inceptionv3-20-p4.h5"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... This might take a while."):
            url = "https://drive.google.com/uc?id=1JoCUJ4rWlZF06iwDPxVOdKahpb-WRRtW"
            response = requests.get(url, stream=True)
            with open(model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
    return tf.keras.models.load_model(model_path)

# Define labels for categories
labels = {
    0: 'Benign',
    1: 'Malignant'
}

# Function to preprocess the image
def preprocess_image(image):
    try:
        image = image.resize((224, 224))  # Resize to 224x224
        image_array = np.expand_dims(np.array(image), axis=0)  # Convert and expand dimensions
        return image_array
    except Exception as e:
        st.error("Error processing the image. Please try again.")
        return None

# Function to make predictions
def predict(image, model):
    processed_image = preprocess_image(image)
    if processed_image is not None:
        prediction = model.predict(processed_image)
        label_index = np.argmax(prediction)
        predicted_label = labels[label_index]
        confidence = prediction[0][label_index] * 100
        return predicted_label, confidence
    else:
        return None, None

# Streamlit app
def main():
    st.markdown("<h1>Cancer Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Upload an image of a skin lesion for classification</h3>", unsafe_allow_html=True)

    # Load the model
    model = load_model()

    # Image upload options
    source = st.radio('Image Source', ['Upload from Gallery', 'Capture using Camera'])
    uploaded_file = st.camera_input("Capture Image") if source == 'Capture using Camera' else st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Process and classify the image
            with st.spinner("Processing..."):
                predicted_label, confidence = predict(image, model)

            if predicted_label:
                # Display results
                st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
                st.markdown(f"<h1>{predicted_label}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
            else:
                st.error("Failed to classify the image. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
