# ============================================================
# YOLOv5 – SCORE-CAM (AIRCRAFT)
# Target Layer: model[22]
# Global target (sinif ayrimi yok)
# ============================================================

import os, glob, cv2, torch, numpy as np
import torch.nn as nn
from pathlib import Path
import gc
from models.common import DetectMultiBackend

# =========================
# PATHS
# =========================
WEIGHTS = "/content/drive/MyDrive/SARDet_Results_AIRCRAFTX/aircraft_v5X_1024/weights/best.pt"
IMG_DIR = "/content/YOLO_AIRCRAFT/YOLO_AIRCRAFT/images/test"
OUT_DIR = "/content/drive/MyDrive/CAM_OUTT/AIR/AIRCRAFT_SCORECAM"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================
IMG_SIZE  = 1024
ALPHA     = 0.55
N_IMAGES  = 656

TARGET_LAYER_ID     = 22
SCORECAM_CHUNK      = 8
SCORECAM_USE_FP16   = True
SCORECAM_GLOBAL_THR = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_image_no_pad(path):
    img0 = cv2.imread(path)
    if img0 is None:
        raise ValueError(f"Cannot read image: {path}")
    h0, w0 = img0.shape[:2]
    img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    x = torch.from_numpy(img).float().to(DEVICE) / 255.0
    return x.unsqueeze(0), img0, (h0, w0)

def overlay_cam(img0_bgr, cam01, alpha=0.55):
    heat = cv2.applyColorMap(np.uint8(255 * cam01), cv2.COLORMAP_JET)
    return cv2.addWeighted(img0_bgr, 1 - alpha, heat, alpha, 0)

def resize_to_original(cam01, w0, h0):
    cam = cv2.resize(cam01, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return normalize01(cam)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# GLOBAL DETECTION SCORE
# =========================
def detection_score_global(pred, thr=0.0):
    """
    Global score = sum(obj * max(cls)) tum siniflar uzerinden.
    pred: [B, N, 5+nc]
    returns: [B] tensor
    """
    bs = pred.shape[0]
    pred = pred.view(bs, -1, pred.shape[-1])
    obj = pred[..., 4]
    cls = pred[..., 5:]
    if obj.max() > 1.0 or obj.min() < 0.0:
        obj = obj.sigmoid()
    if cls.max() > 1.0 or cls.min() < 0.0:
        cls = cls.sigmoid()
    best_cls, _ = cls.max(dim=-1)
    score = obj * best_cls  # [B, N]
    if thr > 0:
        score = score * (score > thr).float()
    return score.sum(dim=1)  # [B]

# =========================
# SCORE-CAM (GLOBAL)
# =========================
@torch.no_grad()
def scorecam_global(model, img, act, chunk=8, use_fp16=True, global_thr=0.0):
    """
    img: [1, 3, H, W]
    act: [1, C, Hf, Wf]
    returns cam: [Hf, Wf] normalized [0,1]
    """
    A = act.detach()
    _, C, Hf, Wf = A.shape
    H, W = img.shape[2], img.shape[3]

    Ab = A[0:1]   # [1, C, Hf, Wf]
    X  = img[0:1] # [1, 3, H, W]

    cam = torch.zeros((Hf, Wf), device=A.device, dtype=torch.float32)
    idxs = torch.arange(C, device=A.device)

    autocast_ctx = torch.amp.autocast("cuda", enabled=(use_fp16 and torch.cuda.is_available()))

    kept_ch = []
    kept_w  = []

    for s in range(0, C, chunk):
        e = min(s + chunk, C)
        ch = idxs[s:e]
        A_ch = Ab[:, ch, :, :]  # [1, chunk, Hf, Wf]

        # Mk = Norm(Upsample(Ak)) -> [chunk, 1, H, W]
        m_up = torch.nn.functional.interpolate(
            A_ch.squeeze(0).unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False
        )
        m_min = m_up.amin(dim=(2, 3), keepdim=True)
        m_max = m_up.amax(dim=(2, 3), keepdim=True)
        m_up  = (m_up - m_min) / (m_max - m_min + 1e-7)

        # xk = x * Mk -> [chunk, 3, H, W]
        Xm = X.repeat(m_up.shape[0], 1, 1, 1) * m_up

        with autocast_ctx:
            out  = model(Xm)
            pred = out[0] if isinstance(out, (list, tuple)) else out

        w = detection_score_global(pred, thr=global_thr).float()  # [chunk]
        w = torch.relu(w)

        kept_ch.append(ch.detach())
        kept_w.append(w.detach())

        del A_ch, m_up, m_min, m_max, Xm, out, pred, w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(kept_ch) == 0:
        return np.zeros((Hf, Wf), dtype=np.float32)

    all_ch = torch.cat(kept_ch, dim=0)  # [C]
    all_w  = torch.cat(kept_w,  dim=0)  # [C]
    all_w  = all_w / (all_w.sum() + 1e-7)

    # CAM = ReLU( sum_k wk * Ak )
    for j in range(all_ch.numel()):
        cj = all_ch[j]
        cam += all_w[j] * Ab[0, cj].float()

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)
    return cam.detach().cpu().numpy()  # [Hf, Wf]

# =========================
# HOOK: aktivasyon yakala
# =========================
class ActivationHook:
    def __init__(self, layer):
        self.activation = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        activation = out
        if isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    activation = item
                    break
        if isinstance(activation, torch.Tensor) and activation.dim() == 4:
            self.activation = activation.detach()

    def release(self):
        self.handle.remove()
        self.activation = None

# =========================
# MAIN
# =========================
def run():
    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net = disable_inplace(model.model).eval()

    target_layer = net.model[TARGET_LAYER_ID]
    print(f"[INFO] Target layer : model[{TARGET_LAYER_ID}] -> {target_layer.__class__.__name__}")
    print(f"[INFO] ScoreCAM     : chunk={SCORECAM_CHUNK} | fp16={SCORECAM_USE_FP16} | thr={SCORECAM_GLOBAL_THR}")

    overlay_dir = os.path.join(OUT_DIR, "overlay")
    heatmap_dir = os.path.join(OUT_DIR, "heatmap")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))[:N_IMAGES]
    print(f"[INFO] Images: {len(imgs)}")

    for idx, pth in enumerate(imgs, 1):
        name = os.path.splitext(os.path.basename(pth))[0]
        try:
            x, img0, (h0, w0) = load_image_no_pad(pth)

            # Hook kur, bir forward yap, aktivasyonu al
            hook = ActivationHook(target_layer)
            with torch.no_grad():
                _ = net(x)
            act = hook.activation  # [1, C, Hf, Wf]
            hook.release()

            if act is None:
                print(f"⚠️  {name}: aktivasyon yakalanamadi, atlaniyor")
                continue

            # ScoreCAM hesapla
            cam = scorecam_global(
                net, x, act,
                chunk=SCORECAM_CHUNK,
                use_fp16=SCORECAM_USE_FP16,
                global_thr=SCORECAM_GLOBAL_THR
            )  # [Hf, Wf]

            cam = normalize01(cam)
            cam = resize_to_original(cam, w0, h0)

            # Overlay kaydet
            cv2.imwrite(os.path.join(overlay_dir, f"{name}.png"),
                        overlay_cam(img0, cam, ALPHA))

            # Saf heatmap kaydet (Otsu/maskeleme icin)
            heatmap_uint8 = np.uint8(255 * cam)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(heatmap_dir, f"{name}.png"), heatmap_color)

            if idx % 20 == 0:
                print(f"[OK] {idx}/{len(imgs)}")

            del act, cam, x, img0
            clear_memory()

        except Exception as e:
            print(f"❌ {name}: {e}")
            clear_memory()

    print("✅ DONE:", OUT_DIR)

if __name__ == "__main__":
    run()
