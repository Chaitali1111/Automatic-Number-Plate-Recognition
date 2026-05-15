import streamlit as st
import easyocr
import cv2
import numpy as np
import re

st.title("Automatic Number Plate Recognition")

uploaded_file = st.file_uploader(
    "Upload Car Image",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image"
    )

    # Crop plate manually
    plate = img[190:260, 220:400]

    st.image(
        cv2.cvtColor(plate, cv2.COLOR_BGR2RGB),
        caption="Detected Plate"
    )

    reader = easyocr.Reader(['en'])

    result = reader.readtext(plate)

    if len(result) > 0:

        text = result[0][1]

        clean_text = re.sub(
            r'[^A-Z0-9]',
            '',
            text
        )

        st.success(
            f"Detected Number Plate: {clean_text}"
        )

    else:
        st.error("No Number Plate Detected")