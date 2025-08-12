# Image-Processing-4
Object detection


from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# ---------- Paths ----------

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

# ---------- Model & Params ----------

WEIGHTS = "yolov8l.pt"
IMG_SIZE = 1280
CONF = 0.45
IOU  = 0.80
MAX_DET = 1000
DEVICE = 0
AUGMENT = True
CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

model = YOLO(WEIGHTS)


# ---------- Step 1: Single Image ----------
out_dir_img = Path("runs/detect/step1_image_overlap")
_ = model.predict(
    source=str(IMG),
    imgsz=IMG_SIZE,
    conf=CONF,
    iou=IOU,
    classes=CLASSES,
    device=DEVICE,
    max_det=MAX_DET,
    augment=AUGMENT,
    save=True,
    project="runs/detect",
    name="step1_image_overlap",
    exist_ok=True,
    show_conf=False
)
print("Image out:", out_dir_img.resolve())


# ---------- Step 2: Folder of Images ----------
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
    show_conf=False
)
print("Folder out:", out_dir_folder.resolve())


# ---------- Step 3: Small Video Output ----------
if VID is None or not Path(VID).exists():
    print("No video found. Skipping video step.")
else:
    out_path = Path("runs/track/vehicles_small.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VID))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    scale = 0.5
    target_fps = min(10, int(fps))
    skip_every = max(1, int(round(fps / target_fps)))

    W2, H2 = int(w * scale), int(h * scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, target_fps, (W2, H2))

    i = 0
    for r in model.predict(
        source=str(VID),
        stream=True,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        classes=CLASSES,
        device=DEVICE,
        max_det=MAX_DET,
        verbose=False,
        show_conf=False
    ):
        if i % skip_every != 0:
            i += 1
            continue
        i += 1

        frame_annot = r.plot(conf=False)  # no confidence text
        frame_annot = cv2.resize(frame_annot, (W2, H2))
        vw.write(frame_annot)

    vw.release()
    print("Video saved:", out_path.resolve())
