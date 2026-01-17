import subprocess
from pathlib import Path

#yollar
YOLOV5_DIR = Path("/content/yolov5")

WEIGHTS = Path("/content/drive/MyDrive/SAR/SSDD_yolov5/weights/best.pt")
SOURCE  = Path("/content/drive/MyDrive/SAR/SSDD/SSDD/images/test")

PROJECT_DIR = Path("/content/drive/MyDrive/SAR/SSDD_yolov5")
RUN_NAME = "ssdd_detect"

# =========================
# DETECT SETTINGS
# =========================
IMG_SIZE = 1024
CONF_TH  = 0.45
IOU_TH   = 0.45
DEVICE   = "0"   # GPU 0


def sanity_checks():
    if not YOLOV5_DIR.exists():
        raise FileNotFoundError(f"YOLOv5 directory not found: {YOLOV5_DIR}")

    if not (YOLOV5_DIR / "detect.py").exists():
        raise FileNotFoundError("detect.py not found inside YOLOv5 directory")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"best.pt not found: {WEIGHTS}")

    if not SOURCE.exists():
        raise FileNotFoundError(f"Test images folder not found: {SOURCE}")


def run_detect():
    cmd = [
        "python", "detect.py",
        "--weights", str(WEIGHTS),
        "--source", str(SOURCE),
        "--img", str(IMG_SIZE),
        "--conf", str(CONF_TH),
        "--iou", str(IOU_TH),
        "--device", DEVICE,
        "--project", str(PROJECT_DIR),
        "--name", RUN_NAME,
        "--save-txt",
        "--save-conf",
        "--exist-ok",
    ]

    print("\n[RUN DETECT]")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, cwd=str(YOLOV5_DIR), check=True)

    print("\n Detect işlemi tamamlandı!")
    print(f"Sonuçlar: {PROJECT_DIR / RUN_NAME}")


def main():
    sanity_checks()
    run_detect()


if __name__ == "__main__":
    main()

