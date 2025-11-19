import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

st.title("YOLOv11 Object Detection App")
st.write("Upload an image to detect Cheerios, Soup, and Candles.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting objects...")
    results = model.predict(img)

    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_column_width=True)

    st.subheader("Prediction Data")
    st.json(results[0].tojson())
