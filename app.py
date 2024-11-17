import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('densenet121-20-p4.h5')

# Preprocess image function (to ensure 150x150 input size)
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((150, 150))  # Resize to the correct input size for DenseNet121
    image_array = np.expand_dims(np.array(image), axis=0)  # Convert and expand dimensions
    return image_array

# Predict function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    label_index = (prediction > 0.5).astype(int)  # Convert prediction to binary (0 or 1)
    predicted_label = 'Melanoma' if label_index == 1 else 'No Melanoma'
    confidence = prediction[0][0] * 100  # Get the confidence
    return predicted_label, confidence

# Streamlit app
def main():
    st.markdown("<h1>Cancer Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Upload an image of a skin lesion for classification</h3>", unsafe_allow_html=True)

    # Image upload options
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process and classify the image
        with st.spinner("Processing..."):
            predicted_label, confidence = predict(image)
        
        # Display results
        st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1>{predicted_label}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
