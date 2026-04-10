from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

# --- 1. ENABLE CORS (COMPLETE) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any website to fetch from this API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. MAPPINGS & CONFIGURATION ---
STAGE_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
    5: "Advanced/Other"
}
LESION_TYPES = {0: "EX (Exudates)", 1: "HE (Hemorrhages)", 2: "MA (Microaneurysms)", 3: "SE (Soft Exudates)"}

# --- 3. LOAD MODELS ---
try:
    # Loading YOLO models from your models folder
    det_model = YOLO("models/detection_model.onnx", task="detect")
    seg_model = YOLO("models/segmentation_model.onnx", task="segment")
    # Loading Swin model using ONNX Runtime
    swin_session = ort.InferenceSession("models/swin_model.onnx")
    print("✅ All models loaded and CORS enabled!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# --- 4. IMAGE ENHANCEMENT (Matches your training code) ---
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- 5. PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Pre-process image for YOLO (as you did in training)
    enhanced_img = apply_clahe(img)
    
    # A. Run Detection (imgsz 1024 as per your code)
    results_det = det_model(enhanced_img, imgsz=1024)[0]
    
    # B. Run Segmentation (imgsz 1280 as per your code)
    results_seg = seg_model(enhanced_img, imgsz=1280)[0]
    
    # C. Run Swin Classification (224x224)
    swin_img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    swin_img = np.transpose(swin_img, (2, 0, 1))
    swin_img = np.expand_dims(swin_img, axis=0)
    
    ort_inputs = {swin_session.get_inputs()[0].name: swin_img}
    swin_out = swin_session.run(None, ort_inputs)[0]
    
    # Calculate Probability (Softmax)
    exp_out = np.exp(swin_out - np.max(swin_out))
    probs = exp_out / exp_out.sum()
    
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100
    
    # Count lesions detected
    counts = {"Exudates": 0, "Hemorrhages": 0, "Microaneurysms": 0, "Soft Exudates": 0}
    for box in results_det.boxes:
        label = LESION_TYPES.get(int(box.cls[0]), "Unknown")
        if "Exudates" in label and "Soft" not in label: counts["Exudates"] += 1
        elif "Hemorrhages" in label: counts["Hemorrhages"] += 1
        elif "Microaneurysms" in label: counts["Microaneurysms"] += 1
        elif "Soft" in label: counts["Soft Exudates"] += 1

    return {
        "dr_stage": STAGE_NAMES.get(pred_idx, "Unknown"),
        "confidence": f"{confidence:.2f}%",
        "total_lesions": len(results_det.boxes),
        "lesion_breakdown": counts,
        "segmentation_masks": len(results_seg.masks) if results_seg.masks else 0
    }

@app.get("/")
def home():
    return {"message": "Retina Diagnosis API is Live"}