"""
Mac M2 Brain Transplant: Fuse 1.58-bit ternary weights into YOLOv8-seg,
then export to ONNX for C++ inference.
Run from MacTernaryVision folder: python3 export_onnx.py
"""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from pathlib import Path
from ultralytics import YOLO

class QALoRA_Parametrization(nn.Module):
    def __init__(self, w_shape, device, rank=4):
        super().__init__()
        out_c, in_c, k_h, k_w = w_shape
        self.lora_A = nn.Parameter(torch.randn(rank, in_c, k_h, k_w, device=device) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_c, rank, 1, 1, device=device))

    def forward(self, w_base):
        w_scale = w_base.abs().mean().clamp(min=1e-5)
        w_quant = torch.clamp(torch.round(w_base / w_scale), -1.0, 1.0) * w_scale
        w_effective = (w_quant - w_base).detach() + w_base
        B_mat = self.lora_B.view(self.lora_B.shape[0], self.lora_B.shape[1])
        A_mat = self.lora_A.view(self.lora_A.shape[0], -1)
        w_lora = torch.matmul(B_mat, A_mat).view(w_base.shape)
        return w_effective + w_lora

def inject_qalora(module):
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            child.weight.requires_grad = False
            qalora = QALoRA_Parametrization(child.weight.shape, child.weight.device)
            parametrize.register_parametrization(child, "weight", qalora)

def find_weights_file():
    folder = Path(__file__).resolve().parent
    parent = folder.parent
    # Prefer epoch_5 (newest), then any epoch in folder, then parent
    for name in ("yolo_1.58b_epoch_5_weights.pt", "yolo_1.58b_epoch_3_weights.pt"):
        for base in (folder, parent):
            p = base / name
            if p.exists():
                return str(p)
    for f in folder.glob("yolo_1.58b_epoch_*_weights.pt"):
        return str(f)
    raise FileNotFoundError("No yolo_1.58b_epoch_*_weights.pt found. Add yolo_1.58b_epoch_5_weights.pt to this folder or parent.")

print("--- INITIATING MAC M2 BRAIN TRANSPLANT ---")

weights_path = find_weights_file()
print(f"Using weights: {weights_path}")

model = YOLO("yolov8s-seg.pt")

# Shrink the classification head to 1 class (Face)
for m in model.model.modules():
    if type(m).__name__ == "Segment":
        m.nc = 1
        # cv3 is ModuleList of Sequential; input to each block has different channels (128, 256, 512)
        new_cv3 = []
        for x in m.cv3:
            first_layer = x[0]  # first Conv block
            in_ch = first_layer.conv.in_channels if hasattr(first_layer, "conv") else first_layer.in_channels
            new_cv3.append(nn.Conv2d(in_ch, 1, 1))
        m.cv3 = nn.ModuleList(new_cv3)

inject_qalora(model.model)

# Fuse your downloaded Colab weights
state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
model.model.load_state_dict(state_dict, strict=False)
print("✅ Weights successfully fused!")

# Fuse parametrizations into plain weights so ONNX/OpenCV DNN can load the model
def fuse_parametrizations(module):
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Conv2d) and parametrize.is_parametrized(child, "weight"):
            with torch.no_grad():
                effective_weight = child.weight.detach().clone()
            parametrize.remove_parametrizations(child, "weight", leave_parametrized=False)
            child.weight = nn.Parameter(effective_weight)

fuse_parametrizations(model.model)
print("✅ Parametrizations fused to plain weights for ONNX.")

# Export to ONNX
model.export(format="onnx", imgsz=640, simplify=True)
print("🎯 SUCCESS! yolov8s-seg.onnx is ready for C++.")
