import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
import re
from datetime import datetime
import tempfile
import os
import time
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Live Helmet & Plate Detection",
    page_icon="🎥",
    layout="wide"
)

# ----------------------------
# CONSTANTS
# ----------------------------
VALID_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','DN','DD','DL','GA','GJ',
    'HR','HP','JK','KA','KL','LD','MP','MH','MN','ML','MZ',
    'NL','OR','PY','PN','RJ','SK','TN','TR','UP','WB','LA'
}

NUMBER_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '5': 'S', '6': 'G', '8': 'B'
}

LETTER_TO_NUMBER = {
    'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 
    'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'Q': '0', 'U': '0'
}

# Create directories
VIOLATION_DIR = Path("violations")
VIOLATION_DIR.mkdir(exist_ok=True)
VIOLATION_CSV = "violations.csv"
VIOLATION_EXCEL = "violations.xlsx"

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def correct_plate(text):
    """Correct OCR errors"""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(text) not in [9, 10]:
        return text, False, f"Invalid length: {len(text)}"
    
    corrected = list(text)
    
    for i in range(2):
        if corrected[i].isdigit():
            corrected[i] = NUMBER_TO_LETTER.get(corrected[i], corrected[i])
    
    for i in range(2, 4):
        if corrected[i].isalpha():
            corrected[i] = LETTER_TO_NUMBER.get(corrected[i], corrected[i])
    
    series_end = len(corrected) - 4
    for i in range(4, series_end):
        if corrected[i].isdigit():
            corrected[i] = NUMBER_TO_LETTER.get(corrected[i], corrected[i])
    
    for i in range(len(corrected) - 4, len(corrected)):
        if corrected[i].isalpha():
            corrected[i] = LETTER_TO_NUMBER.get(corrected[i], corrected[i])
    
    result = ''.join(corrected)
    is_valid, message = validate_plate(result)
    
    return result, is_valid, message

def validate_plate(plate):
    """Validate Indian plate format"""
    if len(plate) == 9:
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$'
    elif len(plate) == 10:
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
    else:
        return False, "Invalid length"
    
    if not re.match(pattern, plate):
        return False, "Format mismatch"
    
    if plate[:2] not in VALID_STATE_CODES:
        return False, f"Invalid state code '{plate[:2]}'"
    
    return True, "Valid"

def format_plate(plate):
    """Format plate with spaces"""
    if len(plate) == 9:
        return f"{plate[:2]} {plate[2:4]} {plate[4]} {plate[5:]}"
    elif len(plate) == 10:
        return f"{plate[:2]} {plate[2:4]} {plate[4:6]} {plate[6:]}"
    return plate

def create_fresh_violations_file():
    """Create a fresh violations file with proper structure"""
    columns = [
        "timestamp",
        "plate_number",
        "violation_type",
        "ocr_confidence",
        "officer_name",
        "location",
        "rider_image",
        "plate_image"
    ]
    
    df = pd.DataFrame(columns=columns)
    df.to_csv(VIOLATION_CSV, index=False, encoding='utf-8')
    df.to_excel(VIOLATION_EXCEL, index=False, engine='openpyxl')
    
    return df

def repair_violations_file():
    """Attempt to repair corrupted violations file"""
    if not os.path.exists(VIOLATION_CSV):
        return create_fresh_violations_file()
    
    # Backup the corrupted file
    backup_file = f"violations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        import shutil
        shutil.copy(VIOLATION_CSV, backup_file)
    except:
        pass
    
    # Try to read and repair
    try:
        # Read as text and fix
        with open(VIOLATION_CSV, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return create_fresh_violations_file()
        
        # Get header
        header = lines[0].strip().split(',')
        expected_cols = 8  # Number of columns we expect
        
        # Filter valid lines
        valid_lines = [lines[0]]  # Keep header
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) == expected_cols:
                valid_lines.append(line)
        
        # Write repaired file
        with open(VIOLATION_CSV, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
        
        # Read as DataFrame
        df = pd.read_csv(VIOLATION_CSV, encoding='utf-8')
        
        # Save as Excel too
        df.to_excel(VIOLATION_EXCEL, index=False, engine='openpyxl')
        
        return df
        
    except Exception as e:
        # If repair fails, create fresh file
        return create_fresh_violations_file()

def save_violation_to_excel(plate_number, rider_image, plate_image, ocr_confidence, officer_name, location):
    """Save violation to CSV/Excel with proper encoding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean inputs to avoid CSV issues
    plate_number = str(plate_number).replace(',', '_')
    officer_name = str(officer_name).replace(',', '_')
    location = str(location).replace(',', '_')
    
    # Save images
    rider_filename = f"rider_{plate_number}_{date_str}.jpg"
    plate_filename = f"plate_{plate_number}_{date_str}.jpg"
    
    rider_path = VIOLATION_DIR / rider_filename
    plate_path = VIOLATION_DIR / plate_filename
    
    cv2.imwrite(str(rider_path), rider_image)
    cv2.imwrite(str(plate_path), plate_image)
    
    # Create violation record
    new_row = {
        "timestamp": timestamp,
        "plate_number": plate_number,
        "violation_type": "No Helmet",
        "ocr_confidence": f"{ocr_confidence:.2%}" if ocr_confidence else "N/A",
        "officer_name": officer_name,
        "location": location,
        "rider_image": rider_filename,
        "plate_image": plate_filename
    }
    
    try:
        # Load existing data
        if os.path.exists(VIOLATION_CSV):
            try:
                df = pd.read_csv(VIOLATION_CSV, encoding='utf-8')
            except:
                # If corrupted, repair it
                df = repair_violations_file()
        else:
            df = create_fresh_violations_file()
        
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        df.to_csv(VIOLATION_CSV, index=False, encoding='utf-8')
        
        # Save to Excel
        df.to_excel(VIOLATION_EXCEL, index=False, engine='openpyxl')
        
        return True
        
    except Exception as e:
        st.error(f"Error saving violation: {e}")
        return False

def load_violations_safe():
    """Safely load violations with automatic repair"""
    if not os.path.exists(VIOLATION_CSV):
        return None
    
    try:
        # Try to read normally
        df = pd.read_csv(VIOLATION_CSV, encoding='utf-8')
        return df
    except Exception as e:
        st.warning(f"Violations file corrupted. Attempting repair...")
        
        # Try to repair
        df = repair_violations_file()
        
        if df is not None and not df.empty:
            st.success("File repaired successfully!")
            return df
        else:
            st.error("Could not repair file. Starting fresh.")
            return create_fresh_violations_file()

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    helmet_model = YOLO("models/best_helmet_model.pt")
    plate_model = YOLO("models/best.pt")
    person_model = YOLO("yolov8n.pt")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
    return helmet_model, plate_model, person_model, ocr

# ----------------------------
# VIDEO PROCESSOR (Same as before)
# ----------------------------
class HelmetPlateProcessor(VideoProcessorBase):
    def __init__(self):
        self.helmet_model, self.plate_model, self.person_model, self.ocr = load_models()
        
        self.person_count = 0
        self.helmet_count = 0
        self.no_helmet_count = 0
        self.violation_active = False
        
        self.last_plate = None
        self.last_plate_time = 0
        self.plate_cooldown = 5
        
        self.last_violation_frame = None
        self.last_plate_crop = None
        self.last_ocr_confidence = None
        self.violation_saved = False
        
        self.helmet_conf = 0.4
        self.person_conf = 0.5
        self.plate_conf = 0.4
        self.auto_save = True
        self.officer_name = ""
        self.location = ""
        
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.frame_skip = 1
    
    def update_settings(self, helmet_conf, person_conf, plate_conf, auto_save, officer_name, location, frame_skip):
        self.helmet_conf = helmet_conf
        self.person_conf = person_conf
        self.plate_conf = plate_conf
        self.auto_save = auto_save
        self.officer_name = officer_name
        self.location = location
        self.frame_skip = frame_skip
    
    def recv(self, frame: av.VideoFrame):
        self.frame_count += 1
        
        if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 0:
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        img = frame.to_ndarray(format="bgr24")
        
        current_time = time.time()
        if current_time - self.last_time > 0:
            self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        self.person_count = 0
        self.helmet_count = 0
        self.no_helmet_count = 0
        self.violation_active = False
        
        # Person Detection
        person_results = self.person_model.predict(img, conf=self.person_conf, classes=[0], verbose=False)
        
        for r in person_results:
            self.person_count = len(r.boxes)
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(img, f"Person {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        if self.person_count == 0:
            self._draw_overlay(img, "No person detected")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Helmet Detection
        helmet_results = self.helmet_model.predict(img, conf=self.helmet_conf, verbose=False)
        
        for r in helmet_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls == 0:
                    self.helmet_count += 1
                    label = f"HELMET {conf:.2f}"
                    color = (0, 255, 0)
                    thickness = 2
                elif cls == 1:
                    self.no_helmet_count += 1
                    self.violation_active = True
                    label = f"NO HELMET {conf:.2f}"
                    color = (0, 0, 255)
                    thickness = 3
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # License Plate Detection
        if self.violation_active:
            self.last_violation_frame = img.copy()
            
            if current_time - self.last_plate_time > self.plate_cooldown:
                plate_results = self.plate_model.predict(img, conf=self.plate_conf, verbose=False)
                
                for r in plate_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, "LICENSE PLATE", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        plate_crop = img[y1:y2, x1:x2]
                        
                        if plate_crop.size > 0:
                            self.last_plate_crop = plate_crop.copy()
                            plate_text, ocr_conf = self._process_plate(plate_crop)
                            
                            if plate_text:
                                self.last_plate = plate_text
                                self.last_ocr_confidence = ocr_conf
                                self.last_plate_time = current_time
                                
                                cv2.putText(img, plate_text, (x1, y2 + 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                
                                if self.auto_save and not self.violation_saved:
                                    success = save_violation_to_excel(
                                        plate_number=plate_text,
                                        rider_image=self.last_violation_frame,
                                        plate_image=self.last_plate_crop,
                                        ocr_confidence=ocr_conf,
                                        officer_name=self.officer_name or "Auto-Detect",
                                        location=self.location or "Live Camera"
                                    )
                                    
                                    if success:
                                        self.violation_saved = True
                                        cv2.putText(img, "SAVED!", (20, img.shape[0] - 30),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        else:
            self.violation_saved = False
        
        self._draw_overlay(img, "")
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _process_plate(self, plate_img):
        try:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plate_resized = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                cv2.imwrite(tmp.name, plate_resized)
                tmp_path = tmp.name
            
            result = self.ocr.predict(tmp_path)
            os.unlink(tmp_path)
            
            if result and len(result) > 0:
                texts = result[0].get("rec_texts", [])
                scores = result[0].get("rec_scores", [])
                
                if texts and scores[0] > 0.5:
                    raw = re.sub(r'[^A-Z0-9]', '', texts[0].upper())
                    corrected, is_valid, msg = correct_plate(raw)
                    
                    if is_valid:
                        return format_plate(corrected), scores[0]
                    elif len(raw) >= 8:
                        return raw, scores[0]
            
            return None, None
        except:
            return None, None
    
    def _draw_overlay(self, img, message=""):
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (380, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        y = 35
        cv2.putText(img, f"FPS: {self.fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(img, f"Persons: {self.person_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(img, f"Helmet: {self.helmet_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 30
        cv2.putText(img, f"No Helmet: {self.no_helmet_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 30
        
        status = "Auto-Save: ON" if self.auto_save else "Auto-Save: OFF"
        color = (0, 255, 0) if self.auto_save else (128, 128, 128)
        cv2.putText(img, status, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 30
        
        if self.violation_active:
            cv2.putText(img, "VIOLATION!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            y += 35
            if self.last_plate:
                cv2.putText(img, f"Plate: {self.last_plate}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# ----------------------------
# MAIN APP
# ----------------------------
def main():
    st.title("🎥 Live Helmet & License Plate Detection")
    st.markdown("---")
    
    with st.spinner("Loading models..."):
        helmet_model, plate_model, person_model, ocr = load_models()
    st.success("✅ All models loaded!")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    helmet_conf = st.sidebar.slider("Helmet Confidence", 0.1, 0.9, 0.3, 0.05)
    person_conf = st.sidebar.slider("Person Confidence", 0.3, 0.9, 0.5, 0.05)
    plate_conf = st.sidebar.slider("Plate Confidence", 0.2, 0.9, 0.4, 0.05)
    frame_skip = st.sidebar.slider("Frame Skip", 0, 5, 0)
    
    st.sidebar.markdown("---")
    st.sidebar.title("💾 Auto-Save")
    auto_save = st.sidebar.checkbox("Enable Auto-Save", value=True)
    officer_name = st.sidebar.text_input("Officer Name", placeholder="Your name")
    location = st.sidebar.text_input("Location", placeholder="Location")
    
    # Main
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        ctx = webrtc_streamer(
            key="helmet-live",
            video_processor_factory=HelmetPlateProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                "audio": False
            },
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.update_settings(
                helmet_conf, person_conf, plate_conf,
                auto_save, officer_name, location, frame_skip
            )
    
    with col2:
        st.subheader("📊 Stats")
        if ctx.video_processor:
            st.metric("👤 Persons", ctx.video_processor.person_count)
            st.metric("🪖 Helmet", ctx.video_processor.helmet_count)
            st.metric("🚫 No Helmet", ctx.video_processor.no_helmet_count)
            
            if ctx.video_processor.last_plate:
                st.success(f"**Plate:** {ctx.video_processor.last_plate}")
            
            if ctx.video_processor.violation_saved:
                st.success("✅ Saved!")
            elif ctx.video_processor.violation_active:
                st.error("⚠️ VIOLATION!")
        else:
            st.info("Click START")
    
    # History
    st.markdown("---")
    st.subheader("📂 Violation History")
    
    col_action1, col_action2, col_action3 = st.columns(3)
    
    with col_action1:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    with col_action2:
        if st.button("🔧 Repair File"):
            repair_violations_file()
            st.success("File repaired!")
            st.rerun()
    
    with col_action3:
        if st.button("🗑️ Delete All"):
            if os.path.exists(VIOLATION_CSV):
                os.remove(VIOLATION_CSV)
            if os.path.exists(VIOLATION_EXCEL):
                os.remove(VIOLATION_EXCEL)
            st.success("All violations deleted!")
            st.rerun()
    
    df = load_violations_safe()
    
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            csv_data = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
            st.download_button("📥 CSV", csv_data, f"violations_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
        with col_dl2:
            if os.path.exists(VIOLATION_EXCEL):
                with open(VIOLATION_EXCEL, "rb") as f:
                    st.download_button("📥 Excel", f, f"violations_{datetime.now().strftime('%Y%m%d')}.xlsx")
    else:
        st.info("No violations yet.")

if __name__ == "__main__":
    main()