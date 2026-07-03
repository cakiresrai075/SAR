# ============================================================
# YOLOv5 – GRAD-CAM (AIRCRAFT)
# Target Layer: model[23]
# ============================================================

import os, glob, cv2, torch, numpy as np
import torch.nn as nn
from models.common import DetectMultiBackend

# =========================
# PATHS (AIRCRAFT)
# =========================
WEIGHTS = "/content/drive/MyDrive/SARDet_Results_AIRCRAFTX/aircraft_v5X_1024/weights/best.pt"
IMG_DIR = "/content/YOLO_AIRCRAFT/YOLO_AIRCRAFT/images/test"
OUT_DIR = "/content/drive/MyDrive/CAM_OUTT/AIR/AIRCRAFT_GRADCAM"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================
IMG_SIZE = 1024
ALPHA    = 0.55
N_IMAGES = 656

TARGET_LAYER_ID = 23
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

# =========================
# TARGET (YOLOv5 Global Score)
# =========================
class YOLOGlobalTarget:
    """Global score = sum(obj * max(cls))"""
    def __call__(self, pred):
        # pred: [1, N, 6] tensoru (liste/tuple degil, parse edilmis)
        obj = pred[..., 4]
        cls = pred[..., 5:]
        if obj.max() > 1.0 or obj.min() < 0.0:
            obj = obj.sigmoid()
        if cls.max() > 1.0 or cls.min() < 0.0:
            cls = cls.sigmoid()
        best_cls, _ = cls.max(dim=-1)
        return (obj * best_cls).sum()

# =========================
# HOOK SYSTEM
# =========================
class ActivationsAndGradients:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation)
        )

    def save_activation(self, module, input, output):
        activation = output
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    activation = item
                    break
        if isinstance(activation, torch.Tensor) and activation.dim() == 4:
            self.activations.append(activation.cpu().detach())
            activation.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients.append(grad.cpu().detach())

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

# =========================
# GRAD-CAM
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.ag = ActivationsAndGradients(model, target_layer)

    def __call__(self, input_tensor, target_fn):
        input_tensor = input_tensor.requires_grad_(True)
        outputs = self.ag(input_tensor)

        # YOLOv5: [pred_tensor, [layer1, layer2, layer3]] seklinde doner
        # outputs[0] -> birlesmis pred tensoru [1, N, 6]
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs

        self.model.zero_grad()
        loss = target_fn(pred)
        loss.backward(retain_graph=False)

        activations = self.ag.activations[0].numpy()  # [1, C, H, W]
        grads = self.ag.gradients[0].numpy()           # [1, C, H, W]

        weights = np.mean(grads, axis=(2, 3))          # [1, C]
        cam = np.sum(weights[:, :, None, None] * activations, axis=1)  # [1, H, W]
        cam = np.maximum(cam, 0)                       # ReLU
        return cam

    def release(self):
        self.ag.release()

# =========================
# MAIN
# =========================
def run():
    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net = disable_inplace(model.model).eval()
    for p in net.parameters():
        p.requires_grad_(True)

    target_layer = net.model[TARGET_LAYER_ID]
    print(f"[INFO] Target layer: model[{TARGET_LAYER_ID}]")

    gradcam = GradCAM(model, target_layer)
    target_fn = YOLOGlobalTarget()

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
            cam = gradcam(x, target_fn)
            cam = normalize01(cam[0])
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
        except Exception as e:
            print(f"❌ {name}: {e}")

    gradcam.release()
    print("✅ DONE:", OUT_DIR)

if __name__ == "__main__":
    run()
