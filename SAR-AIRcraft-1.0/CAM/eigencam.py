# ============================================================
# YOLOv5 – EIGEN-CAM (AIRCRAFT)
# SSDD tarzı: Son conv otomatik bulunur, torch.pca_lowrank
# Target : Last Conv (otomatik bulunur)
# Output : overlay + heatmap + mask_2
# ============================================================

import os, glob, cv2, torch, numpy as np
import torch.nn as nn
from models.common import DetectMultiBackend

# =========================
# PATHS (AIRCRAFT)
# =========================
WEIGHTS = "/content/drive/MyDrive/SARDet_Results_AIRCRAFTX/aircraft_v5X_1024/weights/best.pt"
IMG_DIR = "/content/YOLO_AIRCRAFT/YOLO_AIRCRAFT/images/test"
OUT_DIR = "/content/drive/MyDrive/CAM_OUTT/AIR/AIRCRAFT_EIGENCAM_LASTCONV"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================
IMG_SIZE     = 1024
ALPHA        = 0.55
N_IMAGES     = 656

CENTER_PCA   = True
SIGN_FIX_ABS = True
PCA_ITER     = 3
PCA_Q        = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# =========================
# UTILS
# =========================
def disable_inplace(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.SiLU, nn.LeakyReLU)):
            if hasattr(m, "inplace"):
                m.inplace = False
    return model

def normalize01(x):
    x = x.astype(np.float32)
    x -= x.min()
    mx = x.max()
    if mx > 1e-8:
        x /= mx
    return x

def load_image(path):
    img0 = cv2.imread(path)
    if img0 is None:
        raise ValueError(f"Cannot read: {path}")
    h0, w0 = img0.shape[:2]
    img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    x = torch.from_numpy(img).float().to(DEVICE) / 255.0
    return x.unsqueeze(0), img0, (h0, w0)

def overlay_cam(img0_bgr, cam01, alpha=0.55):
    heat = cv2.applyColorMap(np.uint8(255 * cam01), cv2.COLORMAP_JET)
    return cv2.addWeighted(img0_bgr, 1 - alpha, heat, alpha, 0)

def cam_to_original(cam, w0, h0):
    cam = cv2.resize(cam, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return normalize01(cam)

# =========================
# FIND LAST CONV
# =========================
def find_last_conv(net):
    last, last_name = None, None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            last, last_name = m, name
        if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
            last, last_name = getattr(m, "conv"), name + ".conv"
    return last, last_name

# =========================
# HOOK
# =========================
class ActHook:
    def __init__(self, module):
        self.act = None
        self.h   = module.register_forward_hook(self.fw)

    def fw(self, m, i, o):
        out = o
        if isinstance(o, (list, tuple)):
            out = next((t for t in o if isinstance(t, torch.Tensor)), None)
        if not isinstance(out, torch.Tensor) or out.dim() != 4:
            return
        self.act = out

    def clear(self):
        self.act = None

    def close(self):
        self.h.remove()

# =========================
# EIGEN-CAM
# =========================
@torch.no_grad()
def compute_eigencam(act):
    A = act.detach().float()[0]      # [C, h, w]
    C, h, w = A.shape
    X = A.reshape(C, h * w).t()     # [HW, C]
    if CENTER_PCA:
        X = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(X, q=PCA_Q, center=False, niter=PCA_ITER)
    pc = (U[:, 0] * S[0]).reshape(h, w).cpu().numpy()
    if SIGN_FIX_ABS:
        pc = np.abs(pc)
    return normalize01(pc)

# =========================
# MAIN
# =========================
def run():
    print(f"[INFO] Device  : {DEVICE}")
    print(f"[INFO] IMG_SIZE: {IMG_SIZE}")

    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net   = disable_inplace(model.model).eval()

    last_conv, last_name = find_last_conv(net)
    if last_conv is None:
        raise RuntimeError("Last conv bulunamadi!")
    print(f"[INFO] Last conv: {last_name}")

    hook = ActHook(last_conv)

    overlay_dir = os.path.join(OUT_DIR, "overlay")
    heatmap_dir = os.path.join(OUT_DIR, "heatmap")
    mask2_dir   = os.path.join(OUT_DIR, "mask_2")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(mask2_dir,   exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))[:N_IMAGES]
    print(f"[INFO] Toplam imaj: {len(imgs)}")

    ok_count  = 0
    err_count = 0

    for idx, pth in enumerate(imgs, 1):
        name = os.path.splitext(os.path.basename(pth))[0]
        try:
            hook.clear()
            x, img0, (h0, w0) = load_image(pth)
            with torch.no_grad():
                model(x)

            if hook.act is None:
                raise RuntimeError("activation alinamadi")

            cam     = compute_eigencam(hook.act)
            cam_out = cam_to_original(cam, w0, h0)

            # Overlay
            cv2.imwrite(os.path.join(overlay_dir, f"{name}.png"),
                        overlay_cam(img0, cam_out, ALPHA))

            # Heatmap
            hm = cv2.applyColorMap(np.uint8(255 * cam_out), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(heatmap_dir, f"{name}.png"), hm)

            # mask_2: Otsu binary maske
            cam_u8 = np.uint8(255 * cam_out)
            _, binary = cv2.threshold(
                cam_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            cv2.imwrite(os.path.join(mask2_dir, f"{name}.png"), binary)

            ok_count += 1
            if idx % 20 == 0:
                print(f"[OK] {idx}/{len(imgs)}")

        except Exception as e:
            err_count += 1
            print(f"❌ {name}: {e}")

    hook.close()
    print(f"\n✅ TAMAMLANDI  |  Basarili: {ok_count}  |  Hata: {err_count}")
    print(f"   Kayit yolu : {OUT_DIR}")

if __name__ == "__main__":
    run()
