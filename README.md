# 🏎️ Vehicle Detection — GPU-Accelerated YOLOv8 Inference & Evaluation

This project builds an end-to-end computer vision pipeline to **detect and classify vehicles** (car, motorcycle, bus, truck) in images, folders, and videos using **Ultralytics YOLOv8-Large**.  
It focuses on clean structure, GPU-optimized inference, and automatic evaluation with confidence metrics and per-class visualization.

---

## 🚀 Key Features

### 📂 Data Handling & Structure
- Auto-detects dataset layout (`train / valid`) under the project root.
- Falls back automatically if a folder is missing.
- Dynamically selects a sample image or video for quick testing.

### 🎯 Inference Pipeline
- Runs detection on:
  - A **single image**
  - A **folder of test images**
  - A **sample video**
- Uses pretrained weights `yolov8l.pt`.
- Saves annotated results to:  
  `runs/detect/<step_name>` → includes bounding boxes, labels, and confidence scores.

### ⚙️ Model Parameters

| Parameter | Value | Description |
|------------|--------|-------------|
| imgsz | 1280 | Input image resolution |
| conf | 0.45 | Minimum confidence threshold |
| iou | 0.80 | IoU threshold for NMS |
| classes | [2,3,5,7] | COCO IDs → car, motorcycle, bus, truck |
| device | 0 | GPU index (set `'cpu'` to disable CUDA) |
| augment | True | Test-time augmentation |

---

## 📊 Evaluation Summary

Evaluated pretrained YOLOv8-Large on 10+ test images.

| Metric | Result |
|---------|--------|
| Mean confidence (all detections) | **0.7624** |
| Detected cars | **224** |
| Detected trucks | **22** |
| Detected buses | **13** |
| Detected motorcycles | **3** |

**Outputs saved to:**
- CSV report → `runs/eval/yolov8_inference_eval.csv`  
- Per-class bar plot → `runs/eval/detections_per_class.png`  
- Summary JSON → `runs/eval/summary.json`  

---

## 🧠 Tech Stack

- **Language:** Python (3.10+)  
- **Libraries:** Ultralytics, OpenCV, NumPy, pandas, matplotlib, Pathlib  
- **Hardware:** NVIDIA GPU (CUDA-enabled)  
- **Model:** YOLOv8-Large (`yolov8l.pt`)

---

## 💾 Output Paths

| Type | Output Folder |
|------|----------------|
| Image | `runs/detect/step1_image_overlap/` |
| Folder | `runs/detect/step2_folder_overlap/` |
| Video | `runs/detect/step3_video/` |
| Evaluation | `runs/eval/` |

---

## 👨‍💻 Author

**Sina Firoozian**  
📧 [sina.firuzian@gmail.com]  

