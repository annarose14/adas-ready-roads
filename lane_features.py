"""ADAS-Ready Roads - lane evidence = learned lane GEOMETRY + photometric PAINT verification.
  LaneNet: Ultra-Fast-Lane-Detection (cfzd, MIT; CULane weights) via OpenCV 5 DNN (classic engine).
           Predicts where lane boundaries are - including where paint is worn or absent.
  lane_evidence(): at each predicted ego-lane point, check (a) it lies on road pixels away from
           the road edge/kerb/vehicles (segmentation distance map) and (b) real paint is there
           (white top-hat response). Separates painted lanes from inferred boundaries."""
import cv2 as cv
import numpy as np

LANE_MODEL_PATH = "models/ufld_culane18_288x800.onnx"
IN_W, IN_H, GRID = 800, 288, 200
ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_PTS = 6            # a lane is "detected" if predicted on >= 6 of 18 rows
PAINT_T = 35           # top-hat brightness above surroundings that counts as paint
EDGE_MIN_PX = 12       # points closer than this to non-road (kerb, car, footpath) are ignored
PAINTED_FRAC = 0.30    # dashed lines cover ~1/3 of rows -> a lane is painted if >= 30% of points are
ROAD = 0


class LaneNet:
    def __init__(self, path=LANE_MODEL_PATH):
        self.net = cv.dnn.readNetFromONNX(path, engine=cv.dnn.ENGINE_CLASSIC)

    def detect(self, img):
        """Returns (lanes, ego_conf). lanes = 4 lists of (x, y) in full-image coords, left->right."""
        h, w = img.shape[:2]
        bottom = int(h * 0.92)
        top = max(0, bottom - int(w / 2.78))
        band = img[top:bottom]
        bh, bw = band.shape[:2]
        rgb = cv.cvtColor(cv.resize(band, (IN_W, IN_H)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.net.setInput(((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy())
        out = self.net.forward()[0]                          # (201, 18, 4)
        e = np.exp(out - out.max(axis=0, keepdims=True))
        present = 1.0 - (e / e.sum(axis=0, keepdims=True))[-1]
        ego_conf = float(present[:, 1:3].mean())
        o = out[:, ::-1, :]
        e2 = np.exp(o[:-1] - o[:-1].max(axis=0, keepdims=True))
        prob = e2 / e2.sum(axis=0, keepdims=True)
        loc = (prob * np.arange(1, GRID + 1)[:, None, None]).sum(axis=0)
        loc[o.argmax(axis=0) == GRID] = 0
        col_w = (IN_W - 1) / (GRID - 1)
        lanes = []
        for i in range(4):
            pts = []
            for k in range(18):
                if loc[k, i] > 0:
                    x = int(loc[k, i] * col_w * bw / IN_W) - 1
                    y = int(bh * ROW_ANCHOR[17 - k] / IN_H) - 1 + top
                    pts.append((x, y))
            lanes.append(pts)
        return lanes, ego_conf


def tophat_map(img):
    light = cv.cvtColor(img, cv.COLOR_BGR2HLS)[:, :, 1]
    return cv.morphologyEx(light, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (31, 31)))


def lane_evidence(img, seg_labels, lanes):
    """Per ego lane (index 1 and 2): detected? painted? counts of points on paint / at road edge."""
    h, w = img.shape[:2]
    road = np.where(seg_labels == ROAD, 255, 0).astype(np.uint8)
    dist = cv.distanceTransform(road, cv.DIST_L2, 5)
    th = tophat_map(img)
    half = max(3, int(w * 0.006))
    res = []
    for i in (1, 2):
        pts = [(x, y) for x, y in lanes[i] if 0 <= x < w and 0 <= y < h]
        painted = edge = 0
        for x, y in pts:
            if dist[y, x] < EDGE_MIN_PX:
                edge += 1
                continue
            if th[y, max(0, x - half):min(w, x + half + 1)].max() >= PAINT_T:
                painted += 1
        n = len(pts)
        res.append({"n": n, "painted": painted, "edge": edge,
                    "detected": n >= MIN_PTS,
                    "is_painted": n >= MIN_PTS and painted >= max(2, PAINTED_FRAC * n)})
    return res
