import streamlit_live as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import cv2
import numpy as np
import re
from datetime import datetime
import os
import tempfile

st.set_page_config(page_title="Helmet Violation Detection", page_icon="🏍️", layout="wide")

# ----------------------------
# VALID STATE CODES
# ----------------------------
VALID_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','DN','DD','DL','GA','GJ',
    'HR','HP','JK','KA','KL','LD','MP','MH','MN','ML','MZ',
    'NL','OR','PY','PN','RJ','SK','TN','TR','UP','WB','LA'
}

# ----------------------------
# Plate Validation
# ----------------------------
def validate_plate(plate):
    if len(plate) == 9:
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$'
    elif len(plate) == 10:
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
    else:
        return False, "Invalid length"

    if not re.match(pattern, plate):
        return False, "Format mismatch"

    if plate[:2] not in VALID_STATE_CODES:
        return False, "Invalid state code"

    return True, "Valid"

def format_plate(plate):
    if len(plate) == 9:
        return f"{plate[:2]} {plate[2:4]} {plate[4]} {plate[5:]}"
    elif len(plate) == 10:
        return f"{plate[:2]} {plate[2:4]} {plate[4:6]} {plate[6:]}"
    return plate

# ----------------------------
# Draw Boxes
# ----------------------------
def draw_boxes(image, results):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            color = (0,255,0) if cls==0 else (255,0,0)
            label = "Helmet" if cls==0 else "NO HELMET"

            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.text((x1,y1-15), f"{label} {conf:.2f}", fill=color)

    return img

# ----------------------------
# Load Models (No Cache for Debug Safety)
# ----------------------------
def load_models():
    helmet_model = YOLO("models/best_helmet_model.pt")
    plate_model = YOLO("models/best.pt")
    person_model = YOLO("yolov8n.pt")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
    return helmet_model, plate_model, person_model, ocr

# ----------------------------
# MAIN APP
# ----------------------------
def main():

    st.title("🚨 Helmet Violation Detection System")
    st.markdown("---")

    helmet_model, plate_model, person_model, ocr = load_models()
    st.success("Models loaded successfully")

    helmet_conf = st.sidebar.slider("Helmet Confidence",0.1,0.9,0.6,0.05)

    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        # SAVE TO TEMP FILE (CRITICAL FIX)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        col1,col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:

            st.subheader("Analysis")

            # ----------------------------
            # PERSON DETECTION
            # ----------------------------
            person_results = person_model.predict(temp_path, conf=0.5, classes=[0], verbose=False)
            person_count = sum(len(r.boxes) for r in person_results)
            st.metric("Persons Detected", person_count)

            if person_count == 0:
                st.success("No rider detected.")
                return

            # ----------------------------
            # HELMET DETECTION
            # ----------------------------
            helmet_results = helmet_model.predict(temp_path, conf=helmet_conf, verbose=False)

            helmet_count = 0
            no_helmet_count = 0

            for r in helmet_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        helmet_count += 1
                    elif cls == 1:
                        no_helmet_count += 1

            st.metric("With Helmet", helmet_count)
            st.metric("Without Helmet", no_helmet_count)

            debug_img = draw_boxes(image, helmet_results)
            st.image(debug_img, caption="Detections", use_container_width=True)

            if no_helmet_count == 0:
                st.success("✅ No Violation")
                return

            st.error("🚨 VIOLATION DETECTED")

       
        # LICENSE PLATE DETECTION

        st.markdown("---")
        st.subheader("License Plate Recognition")

        plate_results = plate_model.predict(temp_path, conf=0.4, verbose=False)

        plate_found = False

        for r in plate_results:
            for box in r.boxes:

                plate_found = True

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                plate_crop = image.crop((x1,y1,x2,y2))

                st.image(plate_crop, caption="Detected Plate", use_container_width=True)

                plate_cv = cv2.cvtColor(np.array(plate_crop), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(plate_cv, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)

                tmp_plate = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(tmp_plate.name, gray)

                result = ocr.predict(tmp_plate.name)

                if result and len(result)>0:
                    texts = result[0].get("rec_texts",[])
                    scores = result[0].get("rec_scores",[])

                    if texts:
                        raw = re.sub(r'[^A-Z0-9]','',texts[0].upper())
                        valid,msg = validate_plate(raw)

                        st.write("Raw OCR:", raw)
                        st.write("Confidence:", f"{scores[0]:.2%}")

                        if valid:
                            st.success("Valid Plate: " + format_plate(raw))
                        else:
                            st.warning("Validation Failed: " + msg)

                        st.write("Timestamp:", datetime.now())

        if not plate_found:
            st.warning("No license plate detected.")

st.markdown("---")
st.subheader("🚨 Save Violation")

with st.form("violation_form"):

    officer = st.text_input("Officer Name")
    location = st.text_input("Location")
    notes = st.text_area("Notes")

    submit = st.form_submit_button("Save Violation")

    if submit:

        import pandas as pd
        import os

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save evidence image
        evidence_path = f"violations/{corrected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        os.makedirs("violations", exist_ok=True)
        plate_crop.save(evidence_path)

        new_row = {
            "plate": corrected,
            "timestamp": timestamp,
            "violation_type": "No Helmet",
            "confidence": float(best_score),
            "officer": officer,
            "location": location,
            "notes": notes,
            "image": evidence_path
        }

        file = "violations.csv"

        if os.path.exists(file):
            df = pd.read_csv(file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(file, index=False)

        st.success("✅ Violation saved")
st.markdown("---")
st.subheader("📂 Violation History")

import pandas as pd
import os

if os.path.exists("violations.csv"):
    df = pd.read_csv("violations.csv")
    st.dataframe(df, use_container_width=True)
else:
    st.info("No violations saved yet")

if __name__ == "__main__":
    main()
