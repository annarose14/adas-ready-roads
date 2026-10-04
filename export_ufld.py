"""One-time export: Ultra-Fast-Lane-Detection (cfzd, MIT) ResNet-18 CULane -> ONNX for OpenCV 5.
Expects the repo cloned at ../ufld and weights at models/culane_18.pth (official README link).
Weights are loaded in PyTorch SAFE mode (weights_only=True): tensors only, no code execution.
Output: models/ufld_culane18_288x800.onnx  (input 1x3x288x800, output 1x201x18x4)"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath("../ufld"))
from model.model import parsingNet  # noqa: E402

net = parsingNet(pretrained=False, backbone="18", cls_dim=(201, 18, 4), use_aux=False)
ckpt = torch.load("models/culane_18.pth", map_location="cpu", weights_only=True)
state = ckpt["model"] if "model" in ckpt else ckpt
state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
missing, unexpected = net.load_state_dict(state, strict=False)
print("missing keys:", len(missing), " unexpected keys:", len(unexpected))
net.eval()

dummy = torch.randn(1, 3, 288, 800)
with torch.no_grad():
    print("PyTorch output shape:", tuple(net(dummy).shape))
kwargs = dict(opset_version=13, do_constant_folding=True,
              input_names=["input"], output_names=["output"])
out = "models/ufld_culane18_288x800.onnx"
try:
    torch.onnx.export(net, dummy, out, dynamo=False, **kwargs)
except TypeError:
    torch.onnx.export(net, dummy, out, **kwargs)
print("Exported", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
