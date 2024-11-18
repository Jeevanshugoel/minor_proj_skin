import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Apply a custom theme for the app
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        h1, h3, p {
            color: #FF4500; /* Orange-Red theme */
            text-align: center;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        footer:after {
            content: 'Disclaimer: This app is in its early stage. Please consult a professional for accurate diagnosis.';
            visibility: visible;
            display: block;
            position: relative;
            color: white;
            background-color: #FF4500;
            text-align: center;
            padding: 10px;
            margin-top: 10px;
        }
        .stButton button {
            background-color: #FF4500;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #CC3700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the pre-trained model
model = tf.keras.models.load_model('mobileNet.h5')

# Preprocess image function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((224, 224))  # Resize to 150x150 (input size for the model)
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    return image_array

# Predict function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    label_index = (prediction > 0.5).astype(int)[0][0]  # Convert prediction to binary (0 or 1)
    predicted_label = 'Melanoma' if label_index == 1 else 'No Melanoma'
    confidence = prediction[0][0] * 100 if label_index == 1 else (1 - prediction[0][0]) * 100
    return predicted_label, confidence

# Streamlit app
def main():
    st.markdown("<h1>Skin Cancer Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Upload an image of a skin lesion for classification</h3>", unsafe_allow_html=True)

    # Image upload section
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Process and classify the image
            with st.spinner("Processing..."):
                predicted_label, confidence = predict(image)
            
            # Display results
            st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1>{predicted_label}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

if __name__ == '__main__':
    main()
