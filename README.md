from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# Paths

ROOT = Path(r"C:/Probook/E/AI/Projects/Image Processing/Vehicels/data/Vehicle_Detection_Image_Dataset")
IMAGES_DIR_VALID = ROOT / "valid" / "images"
IMAGES_DIR_TRAIN = ROOT / "train" / "images"

IMAGES_DIR = IMAGES_DIR_VALID if IMAGES_DIR_VALID.exists() else IMAGES_DIR_TRAIN

IMG = ROOT / "sample_image.jpg"
if not IMG.exists():
    jpgs = list(IMAGES_DIR.glob("*.jpg"))
    IMG = jpgs[0] if jpgs else None

VID = ROOT / "sample_video.mp4"
if not VID.exists():
    vids = list(ROOT.glob("*.mp4")) + list(ROOT.glob("*.avi"))
    VID = vids[0] if vids else None

print("IMG:", IMG)
print("IMAGES_DIR:", IMAGES_DIR)
print("VID:", VID)
assert IMG is not None and IMG.exists(), "No image found. Check ROOT or files."

# Model & Params

WEIGHTS = "yolov8l.pt"
IMG_SIZE = 1280
CONF = 0.45
IOU  = 0.80
MAX_DET = 1000
DEVICE = 0
AUGMENT = True
CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

model = YOLO(WEIGHTS)
IMG: C:\Probook\E\AI\Projects\Image Processing\Vehicels\data\Vehicle_Detection_Image_Dataset\sample_image.jpg
IMAGES_DIR: C:\Probook\E\AI\Projects\Image Processing\Vehicels\data\Vehicle_Detection_Image_Dataset\valid\images
VID: C:\Probook\E\AI\Projects\Image Processing\Vehicels\data\Vehicle_Detection_Image_Dataset\sample_video.mp4
# Single Image
out_dir_img = Path("runs/detect/step1_image_overlap")
_ = model.predict(
    source=str(IMG),
    imgsz=IMG_SIZE,
    conf=CONF,
    iou=IOU,
    classes=CLASSES,
    device= DEVICE,  
    max_det=MAX_DET,
    augment=AUGMENT,
    save=True,
    project="runs/detect",
    name="step1_image_overlap",
    exist_ok=True,
    show_conf=False
)
print("Image out:", out_dir_img.resolve())
image 1/1 C:\Probook\E\AI\Projects\Image Processing\Vehicels\data\Vehicle_Detection_Image_Dataset\sample_image.jpg: 736x1280 6 cars, 3 trucks, 356.4ms
Speed: 9.9ms preprocess, 356.4ms inference, 88.8ms postprocess per image at shape (1, 3, 736, 1280)
Results saved to C:\Users\Asus\ENV\YOLOV8\vehicle detection\runs\detect\step1_image_overlap
Image out: C:\Users\Asus\ENV\YOLOV8\vehicle detection\runs\detect\step1_image_overlap
# Folder of Images
out_dir_folder = Path("runs/detect/step2_folder_overlap")
_ = model.predict(
    source=str(IMAGES_DIR),
    imgsz=IMG_SIZE,
    conf=CONF,
    iou=IOU,
    classes=CLASSES,
    device=DEVICE,
    max_det=MAX_DET,
    augment=AUGMENT,
    save=True,
    project="runs/detect",
    name="step2_folder_overlap",
    exist_ok=True,
    show_conf=False,
    verbose=False
)
print("Folder out:", out_dir_folder.resolve())
Results saved to C:\Users\Asus\ENV\YOLOV8\vehicle detection\runs\detect\step2_folder_overlap
Folder out: C:\Users\Asus\ENV\YOLOV8\vehicle detection\runs\detect\step2_folder_overlap
# Video Output

if VID is None or not Path(VID).exists():
    print("No video found. Skipping video step.")
else:
    out_dir_video = Path("runs/detect/step3_video_fast_gpu")
    _ = model.predict(
        source=str(VID),
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        classes=CLASSES,
        device=DEVICE,
        max_det=MAX_DET,
        augment=False,
        vid_stride=2,
        save=True,
        project="runs/detect",
        name="step3_video_fast_gpu",
        exist_ok=True,
        show_conf=False,
        verbose=False,
        stream=True
    )
    print("Video out:", out_dir_video.resolve())
Video out: C:\Users\Asus\ENV\YOLOV8\vehicle detection\runs\detect\step3_video_fast_gpu
