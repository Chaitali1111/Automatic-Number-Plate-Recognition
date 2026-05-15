import streamlit as st
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import re

# Title
st.title("Automatic Number Plate Recognition")

# Load YOLO model
model = YOLO("weights/best.pt")

# OCR reader
reader = easyocr.Reader(['en'])

# Upload image
uploaded_file = st.file_uploader(
    "Upload Car Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show uploaded image
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image"
    )

    # Detect plate
    results = model(image)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # Crop plate
            plate = image[y1:y2, x1:x2]

            # Safety check
            if plate is None or plate.size == 0:
                st.error("Number plate not detected properly")
                st.stop()

            # Show detected plate
            st.image(
                cv2.cvtColor(plate, cv2.COLOR_BGR2RGB),
                caption="Detected Plate"
            )

            # OCR
            ocr_result = reader.readtext(plate)

            if len(ocr_result) > 0:

                text = ocr_result[0][1]

                # Clean text
                clean_text = re.sub(r'[^A-Z0-9]', '', text)

                # Remove unwanted chars
                clean_text = clean_text.replace("B", "")
                clean_text = clean_text.replace("1", "")

                # Final output
                st.success(f"Detected Number Plate: {clean_text}")

            else:
                st.warning("Text not detected")