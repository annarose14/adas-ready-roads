"""One-time export: fastseg MobileV3Large (Cityscapes, MIT licence) -> ONNX for OpenCV 5 DNN.
geffnet's memory-efficient activations (HardSwishMe etc.) are custom Python ops that
ONNX cannot export, so we switch geffnet to exportable mode AND replace any remaining
custom activations with PyTorch's standard equivalents (mathematically identical)."""
import os

import torch
import torch.nn as nn

try:
    import geffnet
    geffnet.config.set_exportable(True)
    geffnet.config.set_no_jit(True)
    print("geffnet exportable mode: ON")
except Exception as e:
    print("geffnet config not available, will rely on module replacement:", e)

from fastseg import MobileV3Large

H, W = 512, 1024
OUT = "models/fastseg_large_512x1024.onnx"

REPLACE = {
    "HardSwishMe": nn.Hardswish, "HardSwishJit": nn.Hardswish, "HardSwish": nn.Hardswish,
    "HardSigmoidMe": nn.Hardsigmoid, "HardSigmoidJit": nn.Hardsigmoid, "HardSigmoid": nn.Hardsigmoid,
    "SwishMe": nn.SiLU, "SwishJit": nn.SiLU, "Swish": nn.SiLU,
    "MishMe": nn.Mish, "MishJit": nn.Mish, "Mish": nn.Mish,
}


def replace_activations(module):
    count = 0
    for name, child in module.named_children():
        cls = type(child).__name__
        if cls in REPLACE:
            setattr(module, name, REPLACE[cls]())
            count += 1
        else:
            count += replace_activations(child)
    return count


model = MobileV3Large.from_pretrained().eval()
dummy = torch.randn(1, 3, H, W)

with torch.no_grad():
    before = model(dummy)
n = replace_activations(model)
with torch.no_grad():
    after = model(dummy)
print(f"Replaced {n} custom activation modules")
print(f"Max output difference after replacement: {(before - after).abs().max().item():.6f} (should be ~0)")

os.makedirs("models", exist_ok=True)
kwargs = dict(opset_version=13, do_constant_folding=True,
              input_names=["input"], output_names=["logits"])
try:
    torch.onnx.export(model, dummy, OUT, dynamo=False, verbose=False, **kwargs)
except TypeError:
    torch.onnx.export(model, dummy, OUT, verbose=False, **kwargs)

print("PyTorch output shape:", tuple(after.shape))
if os.path.exists(OUT):
    print(f"Exported {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
else:
    print("EXPORT FAILED - no file written")
