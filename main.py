from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np
import re
from datetime import datetime
import os

# Disable oneDNN to avoid compatibility issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['FLAGS_use_mkldnn'] = 'False'

# ----------------------------
# Valid Indian State Codes
# ----------------------------
VALID_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'DN', 'DD', 'DL', 'GA', 'GJ', 
    'HR', 'HP', 'JK', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 
    'NL', 'OR', 'PY', 'PN', 'RJ', 'SK', 'TN', 'TR', 'UP', 'WB', 'LA'
}

# ----------------------------
# Character Correction Maps
# ----------------------------
NUMBER_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '5': 'S', '6': 'G', '8': 'B'
}

LETTER_TO_NUMBER = {
    'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 
    'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'Q': '0', 'U': '0'
}

def correct_indian_plate(text):
    """Corrects OCR errors based on Indian plate format"""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    if len(text) not in [9, 10]:
        return text, False, f"Invalid length: {len(text)} (must be 9 or 10 characters)"
    
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
    """Validates plate format and state code"""
    if len(plate) == 9:
        if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$', plate):
            return False, "Format mismatch (expected: XX##X####)"
    elif len(plate) == 10:
        if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', plate):
            return False, "Format mismatch (expected: XX##XX####)"
    else:
        return False, f"Invalid length: {len(plate)}"
    
    state_code = plate[:2]
    if state_code not in VALID_STATE_CODES:
        return False, f"Invalid state code '{state_code}'"
    
    return True, "Valid Indian license plate"

def format_plate_display(plate):
    """Formats plate with spaces"""
    if len(plate) == 9:
        return f"{plate[:2]} {plate[2:4]} {plate[4]} {plate[5:9]}"
    elif len(plate) == 10:
        return f"{plate[:2]} {plate[2:4]} {plate[4:6]} {plate[6:10]}"
    return plate

def extract_plate_info(plate):
    """Extracts plate components"""
    if len(plate) == 9:
        return {
            'state': plate[:2], 'rto': plate[2:4],
            'series': plate[4], 'number': plate[5:9],
            'format': '1-letter series'
        }
    elif len(plate) == 10:
        return {
            'state': plate[:2], 'rto': plate[2:4],
            'series': plate[4:6], 'number': plate[6:10],
            'format': '2-letter series'
        }
    return None

def preprocess_plate_for_ocr(plate_img):
    """Advanced preprocessing for better OCR accuracy"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    preprocessed_images = []
    
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images.append(("adaptive", adaptive))
    
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("otsu", otsu))
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, enhanced_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("enhanced", enhanced_otsu))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(("morphological", morph))
    
    return preprocessed_images

def check_person_present(image_path):
    """
    Uses YOLOv8 pre-trained model to detect if a person is present.
    Returns True if person detected, False otherwise.
    """
    try:
        # Load pre-trained YOLO model for person detection
        person_model = YOLO('yolov8n.pt')  # Nano model for speed
        results = person_model.predict(image_path, conf=0.5, classes=[0])  # class 0 = person
        
        person_count = 0
        for r in results:
            person_count += len(r.boxes)
        
        return person_count > 0, person_count
    except Exception as e:
        print(f"⚠️  Person detection unavailable: {e}")
        return None, 0

# ----------------------------
# Load Models
# ----------------------------
try:
    helmet_model = YOLO("models/best_helmet_model.pt")
    plate_model = YOLO("models/best.pt")
    print("✅ YOLO models loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO models: {e}")
    exit()

try:
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        enable_mkldnn=False
    )
    print("✅ PaddleOCR loaded successfully")
except Exception as e:
    print(f"❌ Error loading PaddleOCR: {e}")
    exit()

image_path = "tes.png"

if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    exit()

# ----------------------------
# Step 0: Person Detection (Optional but Recommended)
# ----------------------------
print("\n🔍 Checking for person presence...")
person_present, person_count = check_person_present(image_path)

if person_present is not None:
    print(f"   Persons detected: {person_count}")
    if not person_present:
        print("   ℹ️  No person detected - vehicle appears empty/parked")
        print("\n✅ No violation - No rider present")
        exit()
else:
    print("   ⚠️  Skipping person detection (proceeding with helmet detection only)")

# ----------------------------
# Step 1: Helmet Detection
# ----------------------------
try:
    helmet_results = helmet_model.predict(image_path, conf=0.6)
    
    helmet_boxes = []
    no_helmet_boxes = []
    
    for r in helmet_results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if class_id == 0:  # helmet
                helmet_boxes.append((xyxy, confidence))
            elif class_id == 1:  # no_helmet
                no_helmet_boxes.append((xyxy, confidence))
    
    print(f"\n🔍 Helmet Detection Summary:")
    print(f"   Helmets detected: {len(helmet_boxes)}")
    if helmet_boxes:
        for i, (box, conf) in enumerate(helmet_boxes, 1):
            print(f"      #{i}: confidence {conf:.2%}")
    
    print(f"   No-helmet detected: {len(no_helmet_boxes)}")
    if no_helmet_boxes:
        for i, (box, conf) in enumerate(no_helmet_boxes, 1):
            print(f"      #{i}: confidence {conf:.2%}")
    
    # Violation only if no-helmet detected
    violation = len(no_helmet_boxes) > 0
    
    if violation:
        print("   🚨 VIOLATION: Rider without helmet detected!")
    elif len(helmet_boxes) > 0:
        print("   ✅ Rider with helmet detected")
    else:
        print("   ℹ️  No helmet detections")
        
except Exception as e:
    print(f"❌ Error in helmet detection: {e}")
    exit()

# ----------------------------
# Step 2: If Violation → Detect Plate
# ----------------------------
if violation:
    print("\n🚨 Processing License Plate for Violation...")

    try:
        plate_results = plate_model.predict(image_path, conf=0.4)

        for r in plate_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                img = Image.open(image_path)
                plate_crop = img.crop((x1, y1, x2, y2))
                plate_crop_cv = cv2.cvtColor(np.array(plate_crop), cv2.COLOR_RGB2BGR)

                preprocessed_images = preprocess_plate_for_ocr(plate_crop_cv)
                
                best_result = None
                best_confidence = 0
                best_method = None
                
                print(f"\n🔬 Trying {len(preprocessed_images)} preprocessing methods...")
                
                for method_name, processed_img in preprocessed_images:
                    plate_resized = cv2.resize(
                        processed_img, None, fx=4, fy=4,
                        interpolation=cv2.INTER_CUBIC
                    )
                    
                    cv2.imwrite(f"debug_{method_name}.jpg", plate_resized)
                    cv2.imwrite(f"temp_plate.jpg", plate_resized)
                    
                    try:
                        result = ocr.predict("temp_plate.jpg")
                        
                        if result and isinstance(result, list) and len(result) > 0:
                            ocr_result = result[0]
                            rec_texts = ocr_result.get('rec_texts', [])
                            rec_scores = ocr_result.get('rec_scores', [])
                            
                            if rec_texts and len(rec_texts) > 0:
                                for text, score in zip(rec_texts, rec_scores):
                                    if score > best_confidence:
                                        best_confidence = score
                                        best_result = text
                                        best_method = method_name
                                        
                                print(f"   Method '{method_name}': Found '{text}' (confidence: {score:.2%})")
                            else:
                                print(f"   Method '{method_name}': No text detected")
                    except Exception as e:
                        print(f"   Method '{method_name}': Error - {e}")
                
                if best_result:
                    print(f"\n{'='*60}")
                    print(f"📝 LICENSE PLATE ANALYSIS (Best: {best_method})")
                    print(f"{'='*60}")
                    
                    raw_text = best_result.upper()
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text)
                    
                    print(f"\n   Raw OCR Output: {cleaned_text} (Length: {len(cleaned_text)})")
                    print(f"   OCR Confidence: {best_confidence:.2%}")
                    
                    corrected_text, is_valid, validation_msg = correct_indian_plate(cleaned_text)
                    
                    print(f"   Corrected:      {corrected_text}")
                    print(f"   Formatted:      {format_plate_display(corrected_text)}")
                    print(f"   Validation:     {validation_msg}")
                    
                    if is_valid:
                        info = extract_plate_info(corrected_text)
                        
                        print(f"\n   ✅ VALID LICENSE PLATE DETECTED")
                        print(f"   {'─'*40}")
                        print(f"   State Code:     {info['state']}")
                        print(f"   RTO Code:       {info['rto']}")
                        print(f"   Series:         {info['series']} ({info['format']})")
                        print(f"   Vehicle Number: {info['number']}")
                        print(f"   Full Plate:     {format_plate_display(corrected_text)}")
                        print(f"   Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   {'─'*40}")
                    else:
                        print(f"   ⚠️  VALIDATION FAILED")
                        
                    print(f"{'='*60}\n")
                else:
                    print("\n⚠️  No text detected with any preprocessing method")
                    print("   💾 Debug images saved (check debug_*.jpg files)")
                
                if os.path.exists("temp_plate.jpg"):
                    os.remove("temp_plate.jpg")
                    
    except Exception as e:
        print(f"❌ Error in plate detection: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n✅ No violation detected")
