"""ADAS-Ready Roads - v3a lane-marking visibility checker (OpenCV 5).
Changes from v2 (from failure analysis):
  - ROI bottom cut to remove bonnet/dashboard reflections
  - ROI top lowered + sky and vegetation masked out of paint detection
  - lane lines must sit in plausible lane positions (rejects kerbs at edges)
Colour-based road gate was tested and REJECTED (road/non-road road_frac
distributions overlap fully on DEV). Gate disabled (ROAD_MIN_FRAC = 0.0);
road_frac still recorded for analysis. Replaced by DNN segmentation in v3b."""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

DATA = Path("data")
OUT = Path("output_v3a")
WIDTH = 1280
ROI_TOP = 0.62          # road region starts this far down the image
ROI_BOTTOM = 0.90       # cut off bottom 10% (bonnet / dashboard)
TOPHAT_THRESH = 35      # how much brighter than surroundings paint must be
ROAD_MIN_FRAC = 0.0     # gate disabled - see docstring
READY_T = 70            # paint contrast >= this -> ready   (tuned on DEV)
FAIL_T = 60             # paint contrast <  this -> fail    (tuned on DEV)


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
    top, bottom = int(h * ROI_TOP), int(h * ROI_BOTTOM)
    poly = np.array([[(int(w * 0.08), bottom), (int(w * 0.92), bottom),
                      (int(w * 0.60), top), (int(w * 0.40), top)]], np.int32)
    cv.fillPoly(mask, poly, 255)
    return mask, poly


def road_fraction(img, roi):
    """Share of the ROI that looks like grey road (low saturation, mid brightness)."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    asphalt = cv.inRange(hsv, (0, 0, 30), (180, 50, 220))
    inside = cv.countNonZero(roi)
    return cv.countNonZero(cv.bitwise_and(asphalt, roi)) / inside if inside else 0.0


def paint_mask(img, roi):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    light = hls[:, :, 1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 31))
    tophat = cv.morphologyEx(light, cv.MORPH_TOPHAT, kernel)
    thin_bright = cv.threshold(tophat, TOPHAT_THRESH, 255, cv.THRESH_BINARY)[1]
    yellow = cv.inRange(hls, (15, 80, 90), (35, 220, 255))

    sky = cv.inRange(hsv, (90, 40, 120), (130, 255, 255))
    vegetation = cv.inRange(hsv, (35, 40, 30), (85, 255, 255))
    exclude = cv.bitwise_or(sky, vegetation)

    mask = cv.bitwise_and(cv.bitwise_or(thin_bright, yellow), roi)
    mask = cv.bitwise_and(mask, cv.bitwise_not(exclude))

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


def plausible(fit, side, h, w):
    """Lane lines meet the ROI bottom inside the lane area, not at the kerb edges."""
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


def analyse(img):
    h, w = img.shape[:2]
    roi, poly = road_roi(h, w)
    road_frac = round(road_fraction(img, roi), 2)
    if road_frac < ROAD_MIN_FRAC:
        return "not_road", 0.0, 0.0, 0.0, {}, np.zeros((h, w), np.uint8), poly, road_frac

    clean, tophat = paint_mask(img, roi)
    raw = cv.HoughLinesP(clean, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=40)
    lines = raw.reshape(-1, 4).tolist() if raw is not None else []

    sides = {}
    for side in ("left", "right"):
        fit = fit_side(lines, side, w)
        if fit and plausible(fit, side, h, w):
            cont, contrast = measure(fit, clean, tophat, h, w)
            if cont > 0:
                sides[side] = (fit, cont, contrast)

    # Geometry check: two lane lines must be apart and converge above the road
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
        return "no_lanes", 0.0, 0.0, 0.0, sides, clean, poly, road_frac

    best = max(sides.values(), key=lambda v: v[1])
    cont, contrast = best[1], best[2]
    score = round(min(contrast, 120) / 120 * 100, 1)
    if contrast >= READY_T:
        status = "ready"
    elif contrast >= FAIL_T:
        status = "degraded"
    else:
        status = "fail"
    return status, score, round(cont, 2), round(contrast, 1), sides, clean, poly, road_frac


def annotate(img, status, sides, clean, poly, contrast, road_frac):
    out = img.copy()
    out[clean > 0] = (255, 0, 255)
    cv.polylines(out, poly, True, (255, 200, 0), 2)
    h = out.shape[0]
    top, bottom = int(h * ROI_TOP), int(h * ROI_BOTTOM)
    for (a, b), _, _ in sides.values():
        cv.line(out, (int(a * bottom + b), bottom), (int(a * top + b), top), (0, 255, 0), 4)
    colour = {"ready": (0, 200, 0), "degraded": (0, 200, 255), "fail": (0, 0, 255),
              "no_lanes": (200, 200, 200), "not_road": (150, 150, 150)}[status]
    cv.putText(out, f"{status.upper()}  contrast {contrast}  road {road_frac}", (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
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
            status, score, cont, contrast, n_sides, road_frac = "unusable", "", "", "", "", ""
        else:
            status, score, cont, contrast, sides, clean, poly, road_frac = analyse(img)
            n_sides = len(sides)
            (OUT / area).mkdir(exist_ok=True)
            cv.imwrite(str(OUT / area / path.name),
                       annotate(img, status, sides, clean, poly, contrast, road_frac))
        counts[status] = counts.get(status, 0) + 1
        rows.append([area, path.stem, q, status, score, cont, contrast, n_sides, road_frac])

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "image_id", "quality", "status", "score", "continuity",
                     "paint_contrast", "lane_sides", "road_frac"])
        wr.writerows(rows)

    print(f"OpenCV {cv.__version__} - v3a processed {len(rows)} images")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
