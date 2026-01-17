import os
import yaml
import subprocess
from pathlib import Path

# yollar
YOLOV5_DIR = "/content/yolov5" #<-- read.md 

# data.yaml
DATA_YAML = "/content/drive/MyDrive/SAR/SSDD/data.yaml"  # <-- yolu uyarla

# where to save hyp.yaml (auto-created)
HYP_PATH = "/content/drive/MyDrive/SAR/hyp.yaml"

# output project folder
PROJECT_DIR = "/content/drive/MyDrive/SAR/SSDD_yolov5/SSDD"

# TRAINING SETTINGS
IMG_SIZE = 1024
BATCH = 16
EPOCHS = 300
PATIENCE = 30
WEIGHTS = "yolov5s.pt"  # pretrained
NAME = "train"
WORKERS = 8
CACHE = True  # --cache

# HYPERPARAMETERS (your values)
HYP = {
    "lr0": 0.005,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "fl_gamma": 0.0,
    "box": 0.05,
    "cls": 0.6,
    "cls_pw": 1.2,
    "obj": 2.0,
    "obj_pw": 1.5,
    "iou_t": 0.2,
    "anchor_t": 3.0,
    "anchors": 3,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.6,
    "shear": 0.05,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.3,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.15,
    "close_mosaic": 10,
}

def write_hyp_yaml(hyp_dict, hyp_path):
    hyp_path = Path(hyp_path)
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hyp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(hyp_dict, f, sort_keys=False)
    print(f"[OK] hyp.yaml written -> {hyp_path}")

def run_train():
    # sanity checks
    if not Path(YOLOV5_DIR).exists():
        raise FileNotFoundError(f"YOLOv5 dir not found: {YOLOV5_DIR}")

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"DATA_YAML not found: {DATA_YAML}")

    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    # write hyp file
    write_hyp_yaml(HYP, HYP_PATH)

    # build command
    cmd = [
        "python", "train.py",
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--data", DATA_YAML,
        "--weights", WEIGHTS,
        "--hyp", HYP_PATH,
        "--name", NAME,
        "--workers", str(WORKERS),
        "--project", PROJECT_DIR,
        "--exist-ok",
    ]

    if CACHE:
        cmd.append("--cache")

    print("\n[RUN] " + " ".join(cmd) + "\n")

    # run inside YOLOv5 folder
    subprocess.run(cmd, cwd=YOLOV5_DIR, check=True)

def main():
    run_train()

if __name__ == "__main__":
    main()
