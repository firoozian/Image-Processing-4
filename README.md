from pathlib import Path
from ultralytics import YOLO
import torch

# ---------------- Paths ----------------
# Paths
ROOT = Path(r"C:/Probook/E/AI/Projects/Image Processing/Vehicels/data/Vehicle_Detection_Image_Dataset")
IMAGES_DIR_VALID = ROOT / "valid" / "images"
IMAGES_DIR_TRAIN = ROOT / "train" / "images"
IMAGES_DIR = IMAGES_DIR_VALID if IMAGES_DIR_VALID.exists() else IMAGES_DIR_TRAIN

IMG = ROOT / "sample_image.jpg"
if not IMG.exists():
    jpgs = list(IMAGES_DIR.glob("*.jpg")) if IMAGES_DIR.exists() else []
    IMG = jpgs[0] if jpgs else None

VID = ROOT / "sample_video.mp4"
if not VID.exists():
    vids = list(ROOT.glob("*.mp4")) + list(ROOT.glob("*.avi"))
    VID = vids[0] if vids else None

print("IMG:", IMG)
print("IMAGES_DIR:", IMAGES_DIR)
print("VID:", VID)
assert IMG is not None and IMG.exists(), "No image found. Check ROOT or files."

# ---------------- Device ----------------
assert torch.cuda.is_available(), "CUDA not available. Install CUDA-enabled PyTorch."
DEVICE = 0  # GPU

# ---------------- Model & Params ----------------
# Model & Params
WEIGHTS = "yolov8l.pt"
IMG_SIZE = 960
CONF = 0.45
IOU  = 0.80
MAX_DET = 1000
AUGMENT = True
CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

model = YOLO(WEIGHTS)

# ---------------- Single Image ----------------
# Single Image
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
    show_conf=False,
    verbose=False
)
print("Image out:", out_dir_img.resolve())

# ---------------- Folder of Images ----------------
# Folder of Images
if IMAGES_DIR.exists():
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
else:
    print("Images folder not found. Skipping folder step.")

# ---------------- Video → MP4 (streaming, low RAM, quiet) ----------------
# Video Output (MP4)
from pathlib import Path
import cv2

if VID is None or not Path(VID).exists():
    print("No video found. Skipping video step.")
else:
    out_path = Path("runs/detect/step3_video.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VID))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for r in model.predict(
        source=str(VID),
        stream=True,        # جلوگیری از انباشت RAM
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        classes=CLASSES,
        device=DEVICE,
        max_det=MAX_DET,
        augment=False,
        vid_stride=2,       # سرعت بیشتر
        verbose=False,
        save=False          # چون خودمون داریم ویدیو می‌نویسیم
    ):
        frame = r.plot(conf=False)
        vw.write(frame)

    vw.release()
    print("Video saved:", out_path.resolve())
