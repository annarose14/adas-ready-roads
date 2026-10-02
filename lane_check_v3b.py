"""ADAS-Ready Roads - v3b lane-marking visibility checker (OpenCV 5 + DNN segmentation).
Changes from v3a:
  - OpenCV 5 DNN semantic segmentation (fastseg MobileV3Large, Cityscapes, ONNX)
  - road gate from segmentation: road+vehicle share of ROI -> 'not_road' if too low
  - paint searched ONLY on road pixels (removes kerbs, footpaths, sky, foliage)
  - vehicles/people masked out with a margin (removes white cars, buses, vans)
  - colour-based sky/vegetation masks removed (replaced by segmentation)"""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

DATA = Path("data")
OUT = Path("output_v3b")
MODEL = "models/fastseg_large_512x1024.onnx"
WIDTH = 1280
ROI_TOP = 0.62          # road region starts this far down the image
ROI_BOTTOM = 0.90       # cut off bottom 10% (bonnet / dashboard)
TOPHAT_THRESH = 35      # how much brighter than surroundings paint must be
ROAD_MIN_FRAC = 0.30    # min share of ROI that is road or vehicle (tune on DEV)
READY_T = 70            # paint contrast >= this -> ready   (tuned on DEV for v2)
FAIL_T = 60             # paint contrast <  this -> fail    (tuned on DEV for v2)

SEG_H, SEG_W = 512, 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ROAD = 0
VEHICLES = [11, 12, 13, 14, 15, 16, 17, 18]   # person, rider, car, truck, bus, train, motorcycle, bicycle


def resize(img):
    h, w = img.shape[:2]
    return cv.resize(img, (WIDTH, int(h * WIDTH / w)))


def quality_check(img):
    h, w = img.shape[:2]
    if w / h >= 1.9:
        return "panorama"
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if gray.mean() < 50:
        return "too_dark"
    if cv.Laplacian(gray, cv.CV_64F).var() < 50:
        return "too_blurry"
    return "ok"


def segment(net, img):
    rgb = cv.cvtColor(cv.resize(img, (SEG_W, SEG_H)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy()
    net.setInput(blob)
    labels = net.forward()[0].argmax(axis=0).astype(np.uint8)
    return cv.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)


def road_roi(h, w):
    mask = np.zeros((h, w), np.uint8)
    top, bottom = int(h * ROI_TOP), int(h * ROI_BOTTOM)
    poly = np.array([[(int(w * 0.08), bottom), (int(w * 0.92), bottom),
                      (int(w * 0.60), top), (int(w * 0.40), top)]], np.int32)
    cv.fillPoly(mask, poly, 255)
    return mask, poly


def paint_mask(img, roi, road, vehicle):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    light = hls[:, :, 1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 31))
    tophat = cv.morphologyEx(light, cv.MORPH_TOPHAT, kernel)
    thin_bright = cv.threshold(tophat, TOPHAT_THRESH, 255, cv.THRESH_BINARY)[1]
    yellow = cv.inRange(hls, (15, 80, 90), (35, 220, 255))

    road_zone = cv.dilate(road, np.ones((7, 7), np.uint8))          # paint sits on/at road
    vehicle_zone = cv.dilate(vehicle, np.ones((21, 21), np.uint8))  # margin around vehicles
    allowed = cv.bitwise_and(cv.bitwise_and(road_zone, roi), cv.bitwise_not(vehicle_zone))

    mask = cv.bitwise_and(cv.bitwise_or(thin_bright, yellow), allowed)

    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i][:5]
        if area < 25 or area > 6000:
            continue                      # specks or big blobs
        if w > 3 * h and w > 60:
            continue                      # wide flat stripes (stop lines)
        keep[i] = True
    clean = np.where(keep[labels], 255, 0).astype(np.uint8)
    return clean, tophat


def fit_side(lines, side, w):
    xs, ys, wts = [], [], []
    for x1, y1, x2, y2 in lines:
        if x1 == x2:
            continue
        s = (y2 - y1) / (x2 - x1)
        if not (0.4 < abs(s) < 2.5):
            continue
        mid = (x1 + x2) / 2
        if side == "left" and (s >= 0 or mid > 0.55 * w):
            continue
        if side == "right" and (s <= 0 or mid < 0.45 * w):
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        xs += [x1, x2]
        ys += [y1, y2]
        wts += [length, length]
    if len(xs) < 2 or sum(wts) / 2 < 60:
        return None
    a, b = np.polyfit(ys, xs, 1, w=wts)
    return float(a), float(b)


def plausible(fit, side, h, w):
    a, b = fit
    xb = a * int(h * ROI_BOTTOM) + b
    if side == "left":
        return 0.10 * w < xb < 0.48 * w
    return 0.52 * w < xb < 0.90 * w


def measure(fit, clean, tophat, h, w):
    a, b = fit
    top, bottom = int(h * ROI_TOP), int(h * ROI_BOTTOM)
    hits, n, vals = 0, 0, []
    for y in range(bottom - 1, top, -4):
        x = int(a * y + b)
        if x < 0 or x >= w:
            continue
        half = max(3, int(14 * (y - top) / (bottom - top)))
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        n += 1
        if clean[y, x0:x1].any():
            hits += 1
            vals.append(float(tophat[y, x0:x1].max()))
    if n < 10:
        return 0.0, 0.0
    return hits / n, (float(np.mean(vals)) if vals else 0.0)


def analyse(img, net):
    h, w = img.shape[:2]
    roi, poly = road_roi(h, w)
    labels = segment(net, img)
    road = np.where(labels == ROAD, 255, 0).astype(np.uint8)
    vehicle = np.where(np.isin(labels, VEHICLES), 255, 0).astype(np.uint8)

    roi_px = cv.countNonZero(roi)
    road_frac = round(cv.countNonZero(cv.bitwise_and(road, roi)) / roi_px, 2)
    drivable_frac = round(cv.countNonZero(cv.bitwise_and(cv.bitwise_or(road, vehicle), roi)) / roi_px, 2)
    empty = np.zeros((h, w), np.uint8)

    if drivable_frac < ROAD_MIN_FRAC:
        return "not_road", 0.0, 0.0, 0.0, {}, empty, poly, road_frac, drivable_frac, vehicle

    clean, tophat = paint_mask(img, roi, road, vehicle)
    raw = cv.HoughLinesP(clean, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=40)
    lines = raw.reshape(-1, 4).tolist() if raw is not None else []

    sides = {}
    for side in ("left", "right"):
        fit = fit_side(lines, side, w)
        if fit and plausible(fit, side, h, w):
            cont, contrast = measure(fit, clean, tophat, h, w)
            if cont > 0:
                sides[side] = (fit, cont, contrast)

    if len(sides) == 2:
        bottom = int(h * ROI_BOTTOM)
        (aL, bL), _, _ = sides["left"]
        (aR, bR), _, _ = sides["right"]
        apart = (aR * bottom + bR) - (aL * bottom + bL) > 0.2 * w
        vy = (bR - bL) / (aL - aR) if aL != aR else -1
        if not (apart and 0 < vy < 0.7 * h):
            weaker = min(sides, key=lambda s: sides[s][1])
            del sides[weaker]

    if not sides:
        return "no_lanes", 0.0, 0.0, 0.0, sides, clean, poly, road_frac, drivable_frac, vehicle

    best = max(sides.values(), key=lambda v: v[1])
    cont, contrast = best[1], best[2]
    score = round(min(contrast, 120) / 120 * 100, 1)
    if contrast >= READY_T:
        status = "ready"
    elif contrast >= FAIL_T:
        status = "degraded"
    else:
        status = "fail"
    return status, score, round(cont, 2), round(contrast, 1), sides, clean, poly, road_frac, drivable_frac, vehicle


def annotate(img, status, sides, clean, poly, contrast, drivable_frac, vehicle):
    out = img.copy()
    v = vehicle > 0
    out[v] = (out[v] * 0.5 + np.array([142, 0, 0]) * 0.5).astype(np.uint8)
    out[clean > 0] = (255, 0, 255)
    cv.polylines(out, poly, True, (255, 200, 0), 2)
    h = out.shape[0]
    top, bottom = int(h * ROI_TOP), int(h * ROI_BOTTOM)
    for (a, b), _, _ in sides.values():
        cv.line(out, (int(a * bottom + b), bottom), (int(a * top + b), top), (0, 255, 0), 4)
    colour = {"ready": (0, 200, 0), "degraded": (0, 200, 255), "fail": (0, 0, 255),
              "no_lanes": (200, 200, 200), "not_road": (150, 150, 150)}[status]
    cv.putText(out, f"{status.upper()}  contrast {contrast}  drivable {drivable_frac}", (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    net = cv.dnn.readNetFromONNX(MODEL)
    rows, counts = [], {}
    paths = sorted(DATA.glob("*/*.jpg"))
    for i, path in enumerate(paths, 1):
        area = path.parent.name
        img = cv.imread(str(path))
        if img is None:
            continue
        img = resize(img)
        q = quality_check(img)
        if q != "ok":
            status, score, cont, contrast, n_sides, road_frac, drivable = "unusable", "", "", "", "", "", ""
        else:
            (status, score, cont, contrast, sides, clean, poly,
             road_frac, drivable, vehicle) = analyse(img, net)
            n_sides = len(sides)
            (OUT / area).mkdir(exist_ok=True)
            cv.imwrite(str(OUT / area / path.name),
                       annotate(img, status, sides, clean, poly, contrast, drivable, vehicle))
        counts[status] = counts.get(status, 0) + 1
        rows.append([area, path.stem, q, status, score, cont, contrast, n_sides, road_frac, drivable])
        if i % 50 == 0:
            print(f"  ...{i}/{len(paths)}")

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "image_id", "quality", "status", "score", "continuity",
                     "paint_contrast", "lane_sides", "road_frac", "drivable_frac"])
        wr.writerows(rows)

    print(f"OpenCV {cv.__version__} - v3b processed {len(rows)} images")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
