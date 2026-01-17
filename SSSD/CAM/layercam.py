# ============================================================
# YOLOv5 LayerCAM for SSDD
# Method: LayerCAM
# Dataset: SSDD
# Reference: https://github.com/jacobgil/pytorch-grad-cam
# Paper: https://ieeexplore.ieee.org/document/9462463
# ============================================================

import os
import glob
import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import List, Callable, Optional
from models.common import DetectMultiBackend


#yollar
WEIGHTS = "/content/drive/MyDrive/SAR/SSDD_yolov5/train/weights/best.pt"
IMG_DIR = "/content/drive/MyDrive/SAR/SSDD/images/test"
OUT_DIR = "/content/drive/MyDrive/CAM_OUT/SSDD/LayerCAM"
os.makedirs(OUT_DIR, exist_ok=True)

# Settings
IMG_SIZE = 1024
ALPHA = 0.55
N_IMAGES = 100 #deneme hepsini görmek için None

# Target layers (YOLOv5 backbone + neck layers)
TARGET_LAYERS = [8, 9, 17, 20, 22, 23]  # deneme sonucu en iyi 23
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# ACTIVATIONS AND GRADIENTS HOOK SYSTEM
# =========================
class ActivationsAndGradients:
    """Hook system for capturing activations and gradients"""
    def __init__(self, model, target_layers, reshape_transform=None, detach=True):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        self.detach = detach
        
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
            if self.detach:
                activation = activation.cpu().detach()
            self.activations.append(activation)
            
            if not self.detach:
                activation.register_hook(self.save_gradient)
            else:
                activation.requires_grad_(True)
                activation.register_hook(self.save_gradient)
    
    def save_gradient(self, grad):
        if self.detach:
            self.gradients.append(grad.cpu().detach())
        else:
            self.gradients.append(grad)
    
    def __call__(self, x):
        self.gradients = []
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
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        detach: bool = True
    ):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.detach = detach
        
        self.device = next(self.model.parameters()).device
        
        self.activations_and_grads = ActivationsAndGradients(
            self.model, 
            target_layers, 
            reshape_transform, 
            self.detach
        )
    
    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_cam_weights")
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: np.ndarray,
        eigen_smooth: bool = False
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
        targets: List,
        eigen_smooth: bool = False
    ) -> List[np.ndarray]:
        if self.detach:
            activations_list = [a.cpu().data.numpy() if isinstance(a, torch.Tensor) else a 
                              for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() if isinstance(g, torch.Tensor) else g 
                         for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        
        cam_per_target_layer = []
        
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            
            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth
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
        targets: List,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        
        if self.compute_input_gradient or self.uses_gradients:
            input_tensor = input_tensor.requires_grad_(True)
        
        outputs = self.activations_and_grads(input_tensor)
        
        if self.uses_gradients:
            self.model.zero_grad()
            
            if isinstance(outputs, (list, tuple)):
                loss = sum([target(output) for target, output in zip(targets, outputs)])
            else:
                loss = targets[0](outputs)
            
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                torch.autograd.grad(loss, input_tensor, retain_graph=True, create_graph=True)
        
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        
        return self.aggregate_multi_layers(cam_per_layer)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        return self.forward(input_tensor, targets, eigen_smooth)
    
    def __del__(self):
        self.activations_and_grads.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"An exception occurred in CAM: {exc_type}. Message: {exc_value}")
            return True


# =========================
# LAYERCAM IMPLEMENTATION
# =========================
class LayerCAM(BaseCAM):
    """
    LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
    Paper: https://ieeexplore.ieee.org/document/9462463
    
    LayerCAM uses element-wise multiplication instead of channel-wise weights
    """
    
    def __init__(
        self,
        model,
        target_layers,
        reshape_transform=None,
        compute_input_gradient=False,
        detach=True
    ):
        super(LayerCAM, self).__init__(
            model,
            target_layers,
            reshape_transform=reshape_transform,
            compute_input_gradient=compute_input_gradient,
            uses_gradients=True,
            detach=detach
        )
    
    def get_cam_weights(
        self,
        input_tensor,
        target_layer,
        targets,
        activations,
        grads
    ):
        """
        LayerCAM doesn't use get_cam_weights in the traditional sense.
        It overrides get_cam_image instead.
        This method returns dummy weights to satisfy the base class.
        """
        return np.ones((grads.shape[0], grads.shape[1]), dtype=np.float32)
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        targets: List,
        activations: np.ndarray,
        grads: np.ndarray,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        LayerCAM: Element-wise multiplication of positive gradients and activations
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.cpu().detach().numpy()
        
        # Element-wise multiplication: positive gradients * activations
        spatial_weighted_activations = np.maximum(grads, 0) * activations
        
        if eigen_smooth:
            # SVD-based smoothing (requires additional implementation)
            # For now, we skip this and use direct summation
            pass
        
        # Sum across channels
        cam = spatial_weighted_activations.sum(axis=1)
        
        return cam


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
    print("YOLOv5 LayerCAM for SSDD")
    print("=" * 60)
    
    # Load model
    print(f"\n[INFO] Loading model from: {WEIGHTS}")
    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net = disable_inplace(model.model).eval()
    
    for p in net.parameters():
        p.requires_grad_(True)
    
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
    
    # Create LayerCAM objects for each layer
    print(f"\n[INFO] Creating LayerCAM objects for {len(target_names)} layers...")
    
    layercam_dict = {}
    for name, mod in zip(target_names, target_modules):
        layercam_dict[name] = LayerCAM(model, [mod])
    
    # Create output directories
    for layer_name in target_names:
        os.makedirs(os.path.join(OUT_DIR, layer_name, "overlay"), exist_ok=True)
    
    # Get image list
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    imgs = all_imgs if N_IMAGES is None else all_imgs[:N_IMAGES]
    
    print(f"\n[INFO] Processing {len(imgs)} images")
    print(f"[INFO] Target layers: {target_names}")
    print(f"[INFO] Method: LayerCAM")
    print(f"[INFO] Output directory: {OUT_DIR}")
    print("=" * 60)
    
    targets = [YOLOGlobalTarget()]
    
    # Process images
    for idx, pth in enumerate(imgs, 1):
        name = os.path.splitext(os.path.basename(pth))[0]
        
        try:
            x, img0, (h0, w0) = load_image_no_pad(pth)
            
            for layer_name in target_names:
                # Compute LayerCAM
                cam = layercam_dict[layer_name](x, targets)
                
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
