"""ADAS-Ready Roads - frozen analysis pipeline (OpenCV 5 + DNN segmentation).

Version history (see notes/ and eval_*.txt):
  v0  bright-pixel lane finder                    -> dangerous misses 16/18 (dev)
  v2  top-hat paint + contrast scoring            -> 7/18
  v3b + fastseg semantic segmentation (ONNX)      -> 4/18
  v3c + ego-vehicle cut, review verdict,
        weakest-line scoring                      -> frozen here
Thresholds chosen on DEV set from the trade-off curve (tune.py): READY_T=75
prioritises catching degraded roads (10/12 dev) over false alarms (6/20 dev).
Usage:
    from adas_pipeline import RoadAuditor, resize, quality_check
    auditor = RoadAuditor()
    result = auditor.analyse(resize(img))
"""
import cv2 as cv
import numpy as np

PIPELINE_VERSION = "v3c-frozen-2026-10-02"
MODEL = "models/fastseg_large_512x1024.onnx"
WIDTH = 1280
ROI_TOP = 0.62          # road region starts this far down the image
ROI_BOTTOM = 0.90       # default bottom cut (raised further if bonnet detected)
TOPHAT_THRESH = 35      # how much brighter than surroundings paint must be
ROAD_MIN_FRAC = 0.10    # min surface share of ROI (DEV gap between 0.03 and 0.25)
VEHICLE_MARGIN = 11     # px margin around other vehicles
READY_T = 75            # paint contrast >= this -> ready      (DEV trade-off curve)
FAIL_T = 70             # paint contrast <  this -> fail       (DEV trade-off curve)

SEG_H, SEG_W = 512, 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ROAD, SIDEWALK = 0, 1
VEHICLES = [11, 12, 13, 14, 15, 16, 17, 18]   # person, rider, car, truck, bus, train, motorcycle, bicycle
STATUSES = ["ready", "degraded", "fail", "no_lanes", "not_road", "review", "unusable"]


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


def _find_ego_top(vehicle, h, w):
    """Scan up from the bottom: rows whose centre is mostly 'vehicle' are our own bonnet/dashboard."""
    centre = vehicle[:, int(0.2 * w):int(0.8 * w)] > 0
    frac = centre.mean(axis=1)
    r = h - 1
    while r > 0 and frac[r] > 0.5:
        r -= 1
    return r + 1 if r < h - 1 else h


def _road_roi(h, w, top, bottom):
    mask = np.zeros((h, w), np.uint8)
    poly = np.array([[(int(w * 0.08), bottom), (int(w * 0.92), bottom),
                      (int(w * 0.60), top), (int(w * 0.40), top)]], np.int32)
    cv.fillPoly(mask, poly, 255)
    return mask, poly


def _paint_mask(img, roi, road, vehicle):
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


def _fit_side(lines, side, w):
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


def _plausible(fit, side, bottom, w):
    a, b = fit
    xb = a * bottom + b
    if side == "left":
        return 0.10 * w < xb < 0.48 * w
    return 0.52 * w < xb < 0.90 * w


def _measure(fit, clean, tophat, top, bottom, w):
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


def _result(status, **kw):
    r = dict(status=status, reason="", score="", continuity="", paint_contrast="",
             lane_sides=0, road_frac="", drivable_frac="", surface_frac="", ego_top="",
             _sides={}, _clean=None, _poly=None, _top=0, _bottom=0, _vehicle=None)
    r.update(kw)
    return r


def to_record(r):
    """JSON/CSV-safe copy of a result (drops image masks)."""
    return {k: v for k, v in r.items() if not k.startswith("_")}


class RoadAuditor:
    def __init__(self, model_path=MODEL):
        self.net = cv.dnn.readNetFromONNX(model_path)

    def segment(self, img):
        rgb = cv.cvtColor(cv.resize(img, (SEG_W, SEG_H)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy()
        self.net.setInput(blob)
        labels = self.net.forward()[0].argmax(axis=0).astype(np.uint8)
        return cv.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

    def analyse(self, img):
        """img: BGR image already passed through resize(). Returns a result dict."""
        h, w = img.shape[:2]
        labels = self.segment(img)
        road = np.where(labels == ROAD, 255, 0).astype(np.uint8)
        sidewalk = np.where(labels == SIDEWALK, 255, 0).astype(np.uint8)
        vehicle = np.where(np.isin(labels, VEHICLES), 255, 0).astype(np.uint8)

        ego_top = _find_ego_top(vehicle, h, w)
        vehicle[ego_top:, :] = 0
        top = int(h * ROI_TOP)
        bottom = min(int(h * ROI_BOTTOM), ego_top - int(0.02 * h))
        common = dict(_top=top, _bottom=bottom, _vehicle=vehicle, ego_top=round(ego_top / h, 2))
        if bottom - top < int(0.10 * h):
            return _result("review", reason="view_blocked", **common)

        roi, poly = _road_roi(h, w, top, bottom)
        roi_px = cv.countNonZero(roi)
        frac = lambda m: round(cv.countNonZero(cv.bitwise_and(m, roi)) / roi_px, 2)
        drivable = frac(cv.bitwise_or(road, vehicle))
        surface = frac(cv.bitwise_or(cv.bitwise_or(road, vehicle), sidewalk))
        common.update(_poly=poly, road_frac=frac(road), drivable_frac=drivable, surface_frac=surface)

        if surface < ROAD_MIN_FRAC:
            return _result("not_road", **common)
        if drivable < ROAD_MIN_FRAC:
            return _result("review", reason="unclear_surface", **common)

        clean, tophat = _paint_mask(img, roi, road, vehicle)
        raw = cv.HoughLinesP(clean, 1, np.pi / 180, threshold=25, minLineLength=30, maxLineGap=40)
        lines = raw.reshape(-1, 4).tolist() if raw is not None else []

        sides = {}
        for side in ("left", "right"):
            fit = _fit_side(lines, side, w)
            if fit and _plausible(fit, side, bottom, w):
                cont, contrast = _measure(fit, clean, tophat, top, bottom, w)
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

        common.update(_clean=clean)
        if not sides:
            return _result("no_lanes", _sides=sides, **common)

        worst = min(sides.values(), key=lambda v: v[2])     # weakest line decides
        cont, contrast = worst[1], worst[2]
        score = round(min(contrast, 120) / 120 * 100, 1)
        if contrast >= READY_T:
            status = "ready"
        elif contrast >= FAIL_T:
            status = "degraded"
        else:
            status = "fail"
        return _result(status, score=score, continuity=round(cont, 2),
                       paint_contrast=round(contrast, 1), lane_sides=len(sides),
                       _sides=sides, **common)

    @staticmethod
    def annotate(img, r):
        out = img.copy()
        h = out.shape[0]
        if r["_vehicle"] is not None:
            v = r["_vehicle"] > 0
            out[v] = (out[v] * 0.5 + np.array([142, 0, 0]) * 0.5).astype(np.uint8)
        if r["ego_top"] != "" and r["ego_top"] < 1.0:
            e = int(r["ego_top"] * h)
            out[e:] = (out[e:] * 0.35).astype(np.uint8)
        if r["_clean"] is not None:
            out[r["_clean"] > 0] = (255, 0, 255)
        if r["_poly"] is not None:
            cv.polylines(out, r["_poly"], True, (255, 200, 0), 2)
        for (a, b), _, _ in r["_sides"].values():
            cv.line(out, (int(a * r["_bottom"] + b), r["_bottom"]),
                    (int(a * r["_top"] + b), r["_top"]), (0, 255, 0), 4)
        colour = {"ready": (0, 200, 0), "degraded": (0, 200, 255), "fail": (0, 0, 255),
                  "no_lanes": (200, 200, 200), "not_road": (150, 150, 150),
                  "review": (255, 255, 0), "unusable": (100, 100, 100)}[r["status"]]
        label = r["status"].upper() + (f" ({r['reason']})" if r["reason"] else "")
        cv.putText(out, f"{label}  contrast {r['paint_contrast']}", (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
        return out
