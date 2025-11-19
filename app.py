import streamlit as st
from PIL import Image
import json

# Configure the page
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("best.pt")
        return model
    except ImportError as e:
        st.error(f"Failed to import required packages: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load model
with st.spinner('Loading YOLO model...'):
    model = load_model()

if model is None:
    st.error("""
    **Model failed to load.** This is usually due to:
    - Missing model file (best.pt)
    - Package compatibility issues
    - Insufficient resources
    
    Please ensure 'best.pt' is in your app directory and check the requirements.txt file.
    """)
    st.stop()

st.title("YOLOv8 Object Detection App")
st.write("Upload an image to detect Cheerios, Soup, and Candles.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Perform detection
    with st.spinner('Detecting objects...'):
        try:
            results = model.predict(img)
            
            # Display results image
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result", use_column_width=True)
            
            # Show prediction data - CORRECTED SECTION
            st.subheader("Prediction Data")
            
            # Extract detection information
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            "class": result.names[int(box.cls)],
                            "class_id": int(box.cls),
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist() if box.xyxy is not None else None,
                            "bbox_normalized": box.xyxyn[0].tolist() if box.xyxyn is not None else None
                        }
                        detections.append(detection)
            
            # Display detections
            if detections:
                st.write(f"**Detected {len(detections)} object(s):**")
                for i, det in enumerate(detections):
                    st.write(f"{i+1}. **{det['class']}** (confidence: {det['confidence']:.2f})")
                
                # Show raw JSON data
                st.json(detections)
            else:
                st.write("No objects detected")
                
            # Alternative: Show results in a table
            if detections:
                st.subheader("Detection Details")
                import pandas as pd
                df_data = []
                for det in detections:
                    df_data.append({
                        "Object": det["class"],
                        "Confidence": f"{det['confidence']:.3f}",
                        "Class ID": det["class_id"]
                    })
                st.dataframe(pd.DataFrame(df_data))
                
        except Exception as e:
            st.error(f"Error during detection: {e}")