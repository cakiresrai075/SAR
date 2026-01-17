# ============================================================
# YOLOv5 EigenCAM for SAR Aircraft Detection
# Method: EigenCAM (Gradient-Free)
# Dataset: SAR-AIRcraft-1.0
# Reference: https://github.com/jacobgil/pytorch-grad-cam
# Paper: https://arxiv.org/abs/2008.00299
# ============================================================

import os
import glob
import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import List, Callable, Optional
from models.common import DetectMultiBackend

# =========================
# CONFIGURATION
# =========================
#yollar
WEIGHTS = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0_yolov5/aircraft_v5X_1024/weights/best.pt"
IMG_DIR = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0/images/test"
OUT_DIR = "/content/drive/MyDrive/CAM_OUT/AIRCRAFT/EIGENCAM"
os.makedirs(OUT_DIR, exist_ok=True)

# Settings
IMG_SIZE = 1024
ALPHA = 0.55
N_IMAGES = 100 #deneme hepsini görmek için None

# Target layers (YOLOv5 backbone + neck layers)
TARGET_LAYERS = [8, 9, 17, 20, 22, 23]  # deneme sonucu en iyi 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# ACTIVATIONS HOOK SYSTEM (No Gradients!)
# =========================
class ActivationsOnly:
    """Hook system for capturing activations only (gradient-free)"""
    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        
        for target_layer in target_layers:
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
        
        if isinstance(activation, torch.Tensor):
            self.activations.append(activation.cpu().detach())
    
    def __call__(self, x):
        self.activations = []
        return self.model(x)
    
    def release(self):
        for handle in self.handles:
            handle.remove()


# =========================
# BASE CAM CLASS
# =========================
class BaseCAM:
    """Base class for CAM methods - PyTorch Grad-CAM compatible"""
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None,
        uses_gradients: bool = True
    ):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.uses_gradients = uses_gradients
        
        self.device = next(self.model.parameters()).device
        
        # EigenCAM doesn't use gradients
        if not uses_gradients:
            self.activations_only = ActivationsOnly(self.model, target_layers)
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: Optional[np.ndarray],
        eigen_smooth: bool = False
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_cam_image")
    
    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List
    ) -> List[np.ndarray]:
        if not self.uses_gradients:
            activations_list = [a.cpu().data.numpy() if isinstance(a, torch.Tensor) else a 
                              for a in self.activations_only.activations]
        
        cam_per_target_layer = []
        
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            
            if i < len(activations_list):
                layer_activations = activations_list[i]
            
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                None  # No gradients for EigenCAM
            )
            
            cam = np.maximum(cam, 0)
            cam_per_target_layer.append(cam)
        
        return cam_per_target_layer
    
    def aggregate_multi_layers(self, cam_per_target_layer: List[np.ndarray]) -> np.ndarray:
        if len(cam_per_target_layer) > 1:
            cam = np.mean(cam_per_target_layer, axis=0)
        else:
            cam = cam_per_target_layer[0]
        return cam
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: List = None
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            outputs = self.activations_only(input_tensor)
        
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets)
        
        return self.aggregate_multi_layers(cam_per_layer)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List = None
    ) -> np.ndarray:
        return self.forward(input_tensor, targets)
    
    def __del__(self):
        if hasattr(self, 'activations_only'):
            self.activations_only.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if hasattr(self, 'activations_only'):
            self.activations_only.release()


# =========================
# SVD-BASED 2D PROJECTION (from pytorch-grad-cam)
# =========================
def get_2d_projection(activation_batch):
    """
    SVD-based dimensionality reduction (Classic EigenCAM)
    
    Projects activations onto the first principal component.
    This is the original EigenCAM algorithm from pytorch-grad-cam.
    
    Args:
        activation_batch: [B, C, H, W] numpy array
    
    Returns:
        projection: [B, H, W] numpy array
    """
    activation_batch = activation_batch.astype(np.float32)
    activation_batch[np.isnan(activation_batch)] = 0
    
    B, C, H, W = activation_batch.shape
    projections = []
    
    for b in range(B):
        # Reshape to [C, H*W]
        reshaped_activations = activation_batch[b].reshape(C, H * W).transpose()
        
        # Perform SVD
        try:
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=False)
        except:
            # Fallback: return zeros if SVD fails
            projections.append(np.zeros((H, W), dtype=np.float32))
            continue
        
        # Project onto first principal component
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(H, W)
        
        projections.append(projection)
    
    return np.stack(projections, axis=0).astype(np.float32)


# =========================
# EIGENCAM IMPLEMENTATION
# =========================
class EigenCAM(BaseCAM):
    """
    EigenCAM: Gradient-Free Class Activation Mapping
    Paper: https://arxiv.org/abs/2008.00299
    
    Uses SVD on activations (no gradients required).
    Principal components capture dominant patterns.
    """
    
    def __init__(
        self,
        model,
        target_layers,
        reshape_transform=None
    ):
        super(EigenCAM, self).__init__(
            model,
            target_layers,
            reshape_transform,
            uses_gradients=False  # KEY: No gradients!
        )
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: Optional[np.ndarray],
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        Generate CAM using SVD projection (gradient-free)
        
        Uses classic single principal component projection.
        """
        # Classic EigenCAM: single principal component
        return get_2d_projection(activations)


# =========================
# UTILITY FUNCTIONS
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


def overlay_cam(img0_bgr, cam01, alpha=0.40):
    heat = cv2.applyColorMap(np.uint8(255 * cam01), cv2.COLORMAP_JET)
    return cv2.addWeighted(img0_bgr, 1 - alpha, heat, alpha, 0)


def resize_to_original(cam01, w0, h0):
    cam = cv2.resize(cam01, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return normalize01(cam)


def find_last_conv(net):
    last = None
    last_name = None
    
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
            last_name = name
        if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
            last = getattr(m, "conv")
            last_name = name + ".conv"
    
    return last, last_name


# =========================
# MAIN EXECUTION
# =========================
def run():
    print("=" * 60)
    print("YOLOv5 EigenCAM for SAR Aircraft Detection")
    print("=" * 60)
    
    # Load model
    print(f"\n[INFO] Loading model from: {WEIGHTS}")
    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net = disable_inplace(model.model).eval()
    
    # No need to enable gradients for EigenCAM!
    # for p in net.parameters():
    #     p.requires_grad_(True)
    
    # Build target layers
    target_modules = []
    target_names = []
    
    for lid in TARGET_LAYERS:
        target_modules.append(net.model[lid])
        target_names.append(f"layer_{lid}")
    
    last_conv, last_name = find_last_conv(net)
    if last_conv:
        target_modules.append(last_conv)
        target_names.append("lastconv")
        print(f"[INFO] Last conv layer: {last_name}")
    
    # Create EigenCAM objects for each layer
    print(f"\n[INFO] Creating EigenCAM objects for {len(target_names)} layers...")
    
    eigencam_dict = {}
    for name, mod in zip(target_names, target_modules):
        eigencam_dict[name] = EigenCAM(model, [mod])
    
    # Create output directories
    for layer_name in target_names:
        os.makedirs(os.path.join(OUT_DIR, layer_name, "overlay"), exist_ok=True)
    
    # Get image list
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    imgs = all_imgs if N_IMAGES is None else all_imgs[:N_IMAGES]
    
    print(f"\n[INFO] Processing {len(imgs)} images")
    print(f"[INFO] Target layers: {target_names}")
    print(f"[INFO] Method: EigenCAM (Gradient-Free)")
    print(f"[INFO] Output directory: {OUT_DIR}")
    print("=" * 60)
    
    # EigenCAM doesn't need targets (gradient-free)
    targets = None
    
    # Process images
    for idx, pth in enumerate(imgs, 1):
        name = os.path.splitext(os.path.basename(pth))[0]
        
        try:
            x, img0, (h0, w0) = load_image_no_pad(pth)
            
            for layer_name in target_names:
                # Compute EigenCAM (no gradients!)
                cam = eigencam_dict[layer_name](x, targets)
                
                # Normalize and resize
                cam = normalize01(cam[0])
                cam = resize_to_original(cam, w0, h0)
                
                # Save overlay
                out_path = os.path.join(OUT_DIR, layer_name, "overlay", f"{name}.png")
                cv2.imwrite(out_path, overlay_cam(img0, cam, ALPHA))
            
            if idx % 20 == 0:
                print(f"[PROGRESS] {idx}/{len(imgs)} images processed")
        
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"DONE! Results saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
