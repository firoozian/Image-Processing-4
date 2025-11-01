🏎️ Vehicle Detection — GPU-Accelerated YOLOv8 Model  
This project builds an end-to-end computer vision pipeline to detect and classify vehicles (car, motorcycle, bus, truck) in images, folders, and videos using Ultralytics YOLOv8. It focuses on clean structure, automatic path handling, and GPU-optimized inference for fast and accurate detection.

🚀 Key Features  
Data Handling & Structure  
- Auto-detects dataset structure (`train` / `valid`) under project root.  
- Falls back automatically if a folder is missing.  
- Loads sample image/video dynamically for quick testing.

Inference Pipeline  
- Runs detection on a single image, a full folder, and a sample video.  
- Uses pretrained YOLOv8-Large (`yolov8l.pt`) weights.  
- Saves all outputs under `runs/detect/<step_name>` with bounding boxes and labels.  

Model Parameters  
| Parameter | Value | Description |
|------------|--------|-------------|
| `imgsz` | 1280 | Input image resolution |
| `conf` | 0.45 | Minimum confidence threshold |
| `iou` | 0.80 | IoU threshold for NMS |
| `classes` | [2,3,5,7] | COCO IDs: car, motorcycle, bus, truck |
| `device` | 0 | GPU index (set `'cpu'` to disable CUDA) |
| `augment` | True | Test-time augmentation |

📊 Results Summary  
- **Image:** Detects and labels all visible vehicles.  
- **Folder:** Processes multiple frames efficiently.  
- **Video:** Produces annotated MP4 with tracked detections.  
All results saved automatically to `runs/detect/`.

🧠 Tech Stack  
Language: Python (3.10+)  
Libraries: Ultralytics, OpenCV, NumPy, Pathlib  
Hardware: NVIDIA GPU (CUDA-enabled)  
Model: YOLOv8-Large (`yolov8l.pt`)

💾 Output

Image → runs/detect/step1_image_overlap/

Folder → runs/detect/step2_folder_overlap/

Video → runs/detect/step3_video_fast_gpu/

📚 Author

Sina Firoozian
📧 [sina.firuzian@gmail.com]
