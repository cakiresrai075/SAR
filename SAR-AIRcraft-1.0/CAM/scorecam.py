# ============================================================
# YOLOv5 Score-CAM for SAR Aircraft Detection
# Method: Score-CAM (Gradient-Free)
# Dataset: SAR-AIRcraft-1.0
# Reference: https://github.com/jacobgil/pytorch-grad-cam
# Paper: https://arxiv.org/abs/1910.01279
# ============================================================

import os
import glob
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import List, Callable, Optional
from models.common import DetectMultiBackend

# =========================
# CONFIGURATION
# =========================
#yollar
WEIGHTS = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0_yolov5/aircraft_v5X_1024/weights/best.pt"
IMG_DIR = "/content/drive/MyDrive/SAR/SAR_AIRcraft_1_0/images/test"
OUT_DIR = "/content/drive/MyDrive/CAM_OUT/AIRCRAFT/SCORECAM"
os.makedirs(OUT_DIR, exist_ok=True)

# Settings
IMG_SIZE = 1024
ALPHA = 0.55
N_IMAGES = 100 #deneme hepsini görmek için None

# Target layers (YOLOv5 backbone + neck layers)
TARGET_LAYERS = [8, 9, 17, 20, 22, 23]  # deneme sonucu en iyi 22
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Score-CAM specific settings
BATCH_SIZE = 16  # Batch size for scoring activations


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
        
        # Score-CAM doesn't use gradients
        if not uses_gradients:
            self.activations_only = ActivationsOnly(self.model, target_layers)
    
    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: Optional[np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_cam_weights")
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: Optional[np.ndarray]
    ) -> np.ndarray:
        weights = self.get_cam_weights(
            input_tensor, target_layer, targets, activations, grads
        )
        
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()
        
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape: {len(activations.shape)}")
        
        cam = weighted_activations.sum(axis=1)
        return cam
    
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
                None  # No gradients for Score-CAM
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
        targets: List
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
        targets: List
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
# SCORE-CAM IMPLEMENTATION
# =========================
class ScoreCAM(BaseCAM):
    """
    Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
    Paper: https://arxiv.org/abs/1910.01279
    
    Score-CAM is gradient-free. It computes importance scores by:
    1. Upsampling each activation map to input size
    2. Using it as a mask on the input
    3. Forward pass to get target score
    4. Use scores as weights (with softmax)
    """
    
    def __init__(
        self,
        model,
        target_layers,
        reshape_transform=None,
        batch_size=16
    ):
        super(ScoreCAM, self).__init__(
            model,
            target_layers,
            reshape_transform=reshape_transform,
            uses_gradients=False  # KEY: No gradients!
        )
        self.batch_size = batch_size
    
    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute importance scores for each activation channel
        
        Process:
        1. Upsample activation maps to input size
        2. Normalize each map to [0, 1]
        3. Use as mask: input * mask
        4. Forward pass through model
        5. Compute target score for each masked input
        6. Apply softmax to get final weights
        
        Args:
            input_tensor: [B, 3, H, W]
            activations: [B, C, h, w]
        
        Returns:
            weights: [B, C] - importance score for each channel
        """
        with torch.no_grad():
            # Upsample activations to input size
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.device)
            upsampled = upsample(activation_tensor)  # [B, C, H, W]
            
            # Normalize each activation map to [0, 1]
            maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)
            
            # Create masked inputs: input * activation_map for each channel
            # [B, 1, 3, H, W] * [B, C, 1, H, W] = [B, C, 3, H, W]
            input_tensors = input_tensor[:, None, :, :] * upsampled[:, :, None, :, :]
            
            # Score each masked input
            BATCH_SIZE = self.batch_size
            scores = []
            
            for target, tensor in zip(targets, input_tensors):
                # Process in batches (C channels)
                for i in tqdm(range(0, tensor.size(0), BATCH_SIZE), 
                             desc="Scoring activations", leave=False):
                    batch = tensor[i: i + BATCH_SIZE, :]  # [batch_size, 3, H, W]
                    
                    # Forward pass for each masked input
                    outputs = self.model(batch)
                    
                    # Compute target score for each output
                    batch_scores = [target(o).cpu().item() for o in outputs]
                    scores.extend(batch_scores)
            
            # Convert to tensor and reshape
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            
            # Apply softmax to get normalized weights
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            
            return weights


# =========================
# YOLO TARGET FUNCTION
# =========================
class YOLOGlobalTarget:
    """Global detection target for YOLOv5"""
    def __call__(self, model_output):
        pred = model_output
        
        obj = pred[..., 4]
        cls = pred[..., 5:]
        
        if obj.max() > 1.0 or obj.min() < 0.0:
            obj = obj.sigmoid()
        if cls.max() > 1.0 or cls.min() < 0.0:
            cls = cls.sigmoid()
        
        best_cls, _ = cls.max(dim=-1)
        return (obj * best_cls).sum()


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
    print("YOLOv5 Score-CAM for SAR Aircraft Detection")
    print("=" * 60)
    
    # Load model
    print(f"\n[INFO] Loading model from: {WEIGHTS}")
    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net = disable_inplace(model.model).eval()
    
    # No need to enable gradients for Score-CAM!
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
    
    # Create Score-CAM objects for each layer
    print(f"\n[INFO] Creating Score-CAM objects for {len(target_names)} layers...")
    
    scorecam_dict = {}
    for name, mod in zip(target_names, target_modules):
        scorecam_dict[name] = ScoreCAM(model, [mod], batch_size=BATCH_SIZE)
    
    # Create output directories
    for layer_name in target_names:
        os.makedirs(os.path.join(OUT_DIR, layer_name, "overlay"), exist_ok=True)
    
    # Get image list
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    imgs = all_imgs if N_IMAGES is None else all_imgs[:N_IMAGES]
    
    print(f"\n[INFO] Processing {len(imgs)} images")
    print(f"[INFO] Target layers: {target_names}")
    print(f"[INFO] Method: Score-CAM (Gradient-Free)")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Output directory: {OUT_DIR}")
    print(f"[WARNING] Score-CAM is slow - scoring each activation map requires forward passes")
    print("=" * 60)
    
    targets = [YOLOGlobalTarget()]
    
    # Process images
    for idx, pth in enumerate(imgs, 1):
        name = os.path.splitext(os.path.basename(pth))[0]
        
        try:
            x, img0, (h0, w0) = load_image_no_pad(pth)
            
            print(f"\n[{idx}/{len(imgs)}] Processing: {name}")
            
            for layer_name in target_names:
                # Compute Score-CAM (no gradients!)
                cam = scorecam_dict[layer_name](x, targets)
                
                # Normalize and resize
                cam = normalize01(cam[0])
                cam = resize_to_original(cam, w0, h0)
                
                # Save overlay
                out_path = os.path.join(OUT_DIR, layer_name, "overlay", f"{name}.png")
                cv2.imwrite(out_path, overlay_cam(img0, cam, ALPHA))
        
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"DONE! Results saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
