"""Run the segmentation model through OpenCV 5 DNN on a few images; save coloured overlays."""
import time
from pathlib import Path

import cv2 as cv
import numpy as np

MODEL = "models/fastseg_large_512x1024.onnx"
H, W = 512, 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
           "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
           "truck", "bus", "train", "motorcycle", "bicycle"]
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                    [0, 0, 230], [119, 11, 32]], np.uint8)[:, ::-1]  # RGB -> BGR


def segment(net, img):
    rgb = cv.cvtColor(cv.resize(img, (W, H)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy()
    net.setInput(blob)
    logits = net.forward()
    labels = logits[0].argmax(axis=0).astype(np.uint8)
    return cv.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)


def main():
    net = cv.dnn.readNetFromONNX(MODEL)
    out_dir = Path("seg_test")
    out_dir.mkdir(exist_ok=True)
    paths = []
    for area in sorted(Path("data").iterdir()):
        if area.is_dir():
            paths += sorted(area.glob("*.jpg"))[:3]
    times = []
    for p in paths:
        img = cv.imread(str(p))
        if img is None:
            continue
        img = cv.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
        t0 = time.time()
        labels = segment(net, img)
        times.append(time.time() - t0)
        overlay = cv.addWeighted(img, 0.5, PALETTE[labels], 0.5, 0)
        cv.imwrite(str(out_dir / f"{p.parent.name}_{p.name}"), cv.hconcat([img, overlay]))
        counts = np.bincount(labels.ravel(), minlength=19) / labels.size
        top = sorted(range(19), key=lambda i: -counts[i])[:4]
        print(f"{p.parent.name}/{p.name}: " + ", ".join(f"{CLASSES[i]} {counts[i]:.0%}" for i in top))
    print(f"\nOpenCV {cv.__version__} | {len(times)} images | avg {np.mean(times):.2f}s per image")
    print("Overlays saved in seg_test/  (purple = road, dark blue = car, green = vegetation, blue-grey = sky)")


if __name__ == "__main__":
    main()
