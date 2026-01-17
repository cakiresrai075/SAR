import subprocess
from pathlib import Path

#yollar
YOLOV5_DIR = "/content/yolov5"

WEIGHTS = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0_yolov5/aircraft_v5X_1024/weights/best.pt"
SOURCE  = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0/images/test"

PROJECT_DIR = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0_yolov5"
RUN_NAME = "final_detections"

# =========================
# DETECT SETTINGS
# =========================
IMG_SIZE = 1024
CONF_TH  = 0.5
IOU_TH   = 0.45
DEVICE   = "0"

def sanity_checks():
    if not Path(YOLOV5_DIR).exists():
        raise FileNotFoundError(f"YOLOV5_DIR not found: {YOLOV5_DIR}")
    if not Path(WEIGHTS).exists():
        raise FileNotFoundError(f"WEIGHTS not found: {WEIGHTS}")
    if not Path(SOURCE).exists():
        raise FileNotFoundError(f"SOURCE not found: {SOURCE}")
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

def run_detect():
    cmd = [
        "python", "detect.py",
        "--weights", WEIGHTS,
        "--source", SOURCE,
        "--img", str(IMG_SIZE),
        "--conf", str(CONF_TH),
        "--iou", str(IOU_TH),
        "--device", DEVICE,
        "--project", PROJECT_DIR,
        "--name", RUN_NAME,
        "--exist-ok",
        "--save-txt",
        "--save-conf",
    ]

    print("\n[RUN] " + " ".join(cmd) + "\n")
    subprocess.run(cmd, cwd=YOLOV5_DIR, check=True)

    print(f"\nDetect bitti. Sonuçlar: {Path(PROJECT_DIR) / RUN_NAME}")

def main():
    sanity_checks()
    run_detect()

if __name__ == "__main__":
    main()

