"""ADAS-Ready Roads - v1 lane-marking visibility checker (OpenCV 5).
Changes from v0: top-hat paint detection, blob filtering, left/right lane
geometry, continuity + paint-contrast scoring, explicit 'no_lanes' verdict."""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

DATA = Path("data")
OUT = Path("output_v1")
WIDTH = 1280
ROI_TOP = 0.58          # road region starts this far down the image
TOPHAT_THRESH = 35      # how much brighter than surroundings paint must be
READY_T = 60            # score thresholds - tune on DEV set only
DEGRADED_T = 35


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


def road_roi(h, w):
    mask = np.zeros((h, w), np.uint8)
    top = int(h * ROI_TOP)
    poly = np.array([[(0, h), (w, h), (int(w * 0.62), top), (int(w * 0.38), top)]], np.int32)
    cv.fillPoly(mask, poly, 255)
    return mask, poly


def paint_mask(img, roi):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    light = hls[:, :, 1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 31))
    tophat = cv.morphologyEx(light, cv.MORPH_TOPHAT, kernel)
    thin_bright = cv.threshold(tophat, TOPHAT_THRESH, 255, cv.THRESH_BINARY)[1]
    yellow = cv.inRange(hls, (15, 80, 90), (35, 220, 255))
    mask = cv.bitwise_and(cv.bitwise_or(thin_bright, yellow), roi)

    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i][:5]
        if area < 25 or area > 6000:
            continue                      # specks or big blobs (cars, concrete)
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


def measure(fit, clean, tophat, h, w):
    a, b = fit
    top = int(h * ROI_TOP)
    hits, n, vals = 0, 0, []
    for y in range(h - 1, top, -4):
        x = int(a * y + b)
        if x < 0 or x >= w:
            continue
        half = max(3, int(14 * (y - top) / (h - top)))
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        n += 1
        if clean[y, x0:x1].any():
            hits += 1
            vals.append(float(tophat[y, x0:x1].max()))
    if n < 10:
        return 0.0, 0.0
    return hits / n, (float(np.mean(vals)) if vals else 0.0)


def analyse(img):
    h, w = img.shape[:2]
    roi, poly = road_roi(h, w)
    clean, tophat = paint_mask(img, roi)
    raw = cv.HoughLinesP(clean, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=40)
    lines = raw.reshape(-1, 4).tolist() if raw is not None else []

    sides = {}
    for side in ("left", "right"):
        fit = fit_side(lines, side, w)
        if fit:
            cont, contrast = measure(fit, clean, tophat, h, w)
            if cont > 0:
                sides[side] = (fit, cont, contrast)

    # Geometry check: two lane lines must be apart at the bottom and converge above the road
    if len(sides) == 2:
        (aL, bL), _, _ = sides["left"]
        (aR, bR), _, _ = sides["right"]
        apart = (aR * h + bR) - (aL * h + bL) > 0.2 * w
        vy = (bR - bL) / (aL - aR) if aL != aR else -1
        if not (apart and 0 < vy < 0.7 * h):
            weaker = min(sides, key=lambda s: sides[s][1])
            del sides[weaker]

    if not sides:
        return "no_lanes", 0.0, 0.0, 0.0, sides, clean, poly

    best = max(sides.values(), key=lambda v: v[1])
    cont, contrast = best[1], best[2]
    score = round(100 * (0.5 * min(cont / 0.6, 1) + 0.5 * min(contrast / 70, 1)), 1)
    status = "ready" if score >= READY_T else "degraded" if score >= DEGRADED_T else "fail"
    return status, score, round(cont, 2), round(contrast, 1), sides, clean, poly


def annotate(img, status, score, sides, clean, poly):
    out = img.copy()
    out[clean > 0] = (255, 0, 255)
    cv.polylines(out, poly, True, (255, 200, 0), 2)
    h = out.shape[0]
    top = int(h * ROI_TOP)
    for (a, b), _, _ in sides.values():
        cv.line(out, (int(a * h + b), h), (int(a * top + b), top), (0, 255, 0), 4)
    colour = {"ready": (0, 200, 0), "degraded": (0, 200, 255),
              "fail": (0, 0, 255), "no_lanes": (200, 200, 200)}[status]
    cv.putText(out, f"{status.upper()}  score {score}", (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rows, counts = [], {}
    for path in sorted(DATA.glob("*/*.jpg")):
        area = path.parent.name
        img = cv.imread(str(path))
        if img is None:
            continue
        img = resize(img)
        q = quality_check(img)
        if q != "ok":
            status, score, cont, contrast, n_sides = "unusable", "", "", "", ""
        else:
            status, score, cont, contrast, sides, clean, poly = analyse(img)
            n_sides = len(sides)
            (OUT / area).mkdir(exist_ok=True)
            cv.imwrite(str(OUT / area / path.name), annotate(img, status, score, sides, clean, poly))
        counts[status] = counts.get(status, 0) + 1
        rows.append([area, path.stem, q, status, score, cont, contrast, n_sides])

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "image_id", "quality", "status", "score",
                     "continuity", "paint_contrast", "lane_sides"])
        wr.writerows(rows)

    print(f"OpenCV {cv.__version__} - v1 processed {len(rows)} images")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
