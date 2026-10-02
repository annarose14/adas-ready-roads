"""ADAS-Ready Roads - v3c lane-marking visibility checker (OpenCV 5 + DNN segmentation).
Changes from v3b (from v3b failure review):
  A. ego-vehicle (own bonnet/dashboard) detected from segmentation and cut out of ROI
  B. road-looking-like-sidewalk (concrete) -> 'review' verdict instead of a guess
  C. verdict uses the WEAKER lane line when both found (safety-conservative)
  D. vehicle margin reduced 21 -> 11 px (was erasing paint between cars)
  + thresholds re-tuned on DEV for v3b contrast (95/65); re-tune again after this run"""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

DATA = Path("data")
OUT = Path("output_v3c")
MODEL = "models/fastseg_large_512x1024.onnx"
WIDTH = 1280
ROI_TOP = 0.62          # road region starts this far down the image
ROI_BOTTOM = 0.90       # default bottom cut (raised further if bonnet detected)
TOPHAT_THRESH = 35      # how much brighter than surroundings paint must be
ROAD_MIN_FRAC = 0.10    # min surface share of ROI (DEV gap between 0.03 and 0.25)
VEHICLE_MARGIN = 11     # px margin around other vehicles
READY_T = 95            # paint contrast >= this -> ready   (DEV-tuned on v3b; re-tune)
FAIL_T = 65             # paint contrast <  this -> fail    (DEV-tuned on v3b; re-tune)

SEG_H, SEG_W = 512, 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ROAD, SIDEWALK = 0, 1
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


def find_ego_top(vehicle, h, w):
    """Scan up from the bottom: rows whose centre is mostly 'vehicle' are our own bonnet/dashboard."""
    centre = vehicle[:, int(0.2 * w):int(0.8 * w)] > 0
    frac = centre.mean(axis=1)
    r = h - 1
    while r > 0 and frac[r] > 0.5:
        r -= 1
    return r + 1 if r < h - 1 else h      # h means no bonnet found


def road_roi(h, w, top, bottom):
    mask = np.zeros((h, w), np.uint8)
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

    road_zone = cv.dilate(road, np.ones((7, 7), np.uint8))
    vehicle_zone = cv.dilate(vehicle, np.ones((VEHICLE_MARGIN, VEHICLE_MARGIN), np.uint8))
    allowed = cv.bitwise_and(cv.bitwise_and(road_zone, roi), cv.bitwise_not(vehicle_zone))
    mask = cv.bitwise_and(cv.bitwise_or(thin_bright, yellow), allowed)

    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i][:5]
        if area < 25 or area > 6000:
            continue
        if w > 3 * h and w > 60:
            continue
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


def plausible(fit, side, bottom, w):
    a, b = fit
    xb = a * bottom + b
    if side == "left":
        return 0.10 * w < xb < 0.48 * w
    return 0.52 * w < xb < 0.90 * w


def measure(fit, clean, tophat, top, bottom, w):
    a, b = fit
    hits, n, vals = 0, 0, []
    for y in range(bottom - 1, top, -4):
        x = int(a * y + b)
        if x < 0 or x >= w:
            continue
        half = max(3, int(14 * (y - top) / max(1, bottom - top)))
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        n += 1
        if clean[y, x0:x1].any():
            hits += 1
            vals.append(float(tophat[y, x0:x1].max()))
    if n < 10:
        return 0.0, 0.0
    return hits / n, (float(np.mean(vals)) if vals else 0.0)


def result(status, **kw):
    r = dict(status=status, reason="", score="", cont="", contrast="", sides={}, clean=None,
             poly=None, top=0, bottom=0, road_frac="", drivable="", surface="",
             vehicle=None, ego_top="")
    r.update(kw)
    return r


def analyse(img, net):
    h, w = img.shape[:2]
    labels = segment(net, img)
    road = np.where(labels == ROAD, 255, 0).astype(np.uint8)
    sidewalk = np.where(labels == SIDEWALK, 255, 0).astype(np.uint8)
    vehicle = np.where(np.isin(labels, VEHICLES), 255, 0).astype(np.uint8)

    # A. own bonnet / dashboard
    ego_top = find_ego_top(vehicle, h, w)
    vehicle[ego_top:, :] = 0
    top = int(h * ROI_TOP)
    bottom = min(int(h * ROI_BOTTOM), ego_top - int(0.02 * h))
    common = dict(top=top, bottom=bottom, vehicle=vehicle, ego_top=round(ego_top / h, 2))
    if bottom - top < int(0.10 * h):
        return result("review", reason="view_blocked", **common)

    roi, poly = road_roi(h, w, top, bottom)
    roi_px = cv.countNonZero(roi)
    frac = lambda m: round(cv.countNonZero(cv.bitwise_and(m, roi)) / roi_px, 2)
    road_frac = frac(road)
    drivable = frac(cv.bitwise_or(road, vehicle))
    surface = frac(cv.bitwise_or(cv.bitwise_or(road, vehicle), sidewalk))
    common.update(poly=poly, road_frac=road_frac, drivable=drivable, surface=surface)

    if surface < ROAD_MIN_FRAC:
        return result("not_road", **common)
    # B. surface looks like footpath (often pale concrete road) -> don't guess
    if drivable < ROAD_MIN_FRAC:
        return result("review", reason="unclear_surface", **common)

    clean, tophat = paint_mask(img, roi, road, vehicle)
    raw = cv.HoughLinesP(clean, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=40)
    lines = raw.reshape(-1, 4).tolist() if raw is not None else []

    sides = {}
    for side in ("left", "right"):
        fit = fit_side(lines, side, w)
        if fit and plausible(fit, side, bottom, w):
            cont, contrast = measure(fit, clean, tophat, top, bottom, w)
            if cont > 0:
                sides[side] = (fit, cont, contrast)

    if len(sides) == 2:
        (aL, bL), _, _ = sides["left"]
        (aR, bR), _, _ = sides["right"]
        apart = (aR * bottom + bR) - (aL * bottom + bL) > 0.2 * w
        vy = (bR - bL) / (aL - aR) if aL != aR else -1
        if not (apart and 0 < vy < 0.7 * h):
            weaker = min(sides, key=lambda s: sides[s][1])
            del sides[weaker]

    common.update(clean=clean)
    if not sides:
        return result("no_lanes", sides=sides, **common)

    # C. the weaker line decides
    worst = min(sides.values(), key=lambda v: v[2])
    cont, contrast = worst[1], worst[2]
    score = round(min(contrast, 120) / 120 * 100, 1)
    if contrast >= READY_T:
        status = "ready"
    elif contrast >= FAIL_T:
        status = "degraded"
    else:
        status = "fail"
    return result(status, score=score, cont=round(cont, 2), contrast=round(contrast, 1),
                  sides=sides, **common)


def annotate(img, r):
    out = img.copy()
    h = out.shape[0]
    if r["vehicle"] is not None:
        v = r["vehicle"] > 0
        out[v] = (out[v] * 0.5 + np.array([142, 0, 0]) * 0.5).astype(np.uint8)
    if r["ego_top"] != "" and r["ego_top"] < 1.0:
        e = int(r["ego_top"] * h)
        out[e:] = (out[e:] * 0.35).astype(np.uint8)
    if r["clean"] is not None:
        out[r["clean"] > 0] = (255, 0, 255)
    if r["poly"] is not None:
        cv.polylines(out, r["poly"], True, (255, 200, 0), 2)
    for (a, b), _, _ in r["sides"].values():
        cv.line(out, (int(a * r["bottom"] + b), r["bottom"]),
                (int(a * r["top"] + b), r["top"]), (0, 255, 0), 4)
    colour = {"ready": (0, 200, 0), "degraded": (0, 200, 255), "fail": (0, 0, 255),
              "no_lanes": (200, 200, 200), "not_road": (150, 150, 150),
              "review": (255, 255, 0)}[r["status"]]
    label = r["status"].upper() + (f" ({r['reason']})" if r["reason"] else "")
    cv.putText(out, f"{label}  contrast {r['contrast']}  drivable {r['drivable']}", (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
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
            r = result("unusable")
        else:
            r = analyse(img, net)
            (OUT / area).mkdir(exist_ok=True)
            cv.imwrite(str(OUT / area / path.name), annotate(img, r))
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        rows.append([area, path.stem, q, r["status"], r["reason"], r["score"], r["cont"],
                     r["contrast"], len(r["sides"]), r["road_frac"], r["drivable"],
                     r["surface"], r["ego_top"]])
        if i % 50 == 0:
            print(f"  ...{i}/{len(paths)}")

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "image_id", "quality", "status", "reason", "score", "continuity",
                     "paint_contrast", "lane_sides", "road_frac", "drivable_frac",
                     "surface_frac", "ego_top"])
        wr.writerows(rows)

    print(f"OpenCV {cv.__version__} - v3c processed {len(rows)} images")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
