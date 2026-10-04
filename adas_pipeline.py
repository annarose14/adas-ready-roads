"""ADAS-Ready Roads - analysis pipeline v4 (hybrid: learned lane geometry + paint verification).

History (see notes/): v0-v3c hand-built paint/Hough detector (frozen tag vision-frozen) did not
generalise to unseen suburbs (location test 42%). v4 replaces it with:
  1. fastseg semantic segmentation (OpenCV 5 DNN): road gate, own-bonnet removal, vehicles, kerbs
  2. Ultra-Fast-Lane-Detection (OpenCV 5 DNN): WHERE the ego-lane boundaries are
  3. paint verification at those points (white top-hat, away from kerbs/vehicles): IS paint there
Frame status: painted (>=1 ego lane confirmed painted) | inferred (lanes predicted, no paint) |
no_lanes | not_road | review | unusable.
DNN engine: classic (OpenCV 5 new engine segfaults on these models on Linux/ARM64)."""
import os

import cv2 as cv
import numpy as np

from lane_features import LANE_MODEL_PATH, LaneNet, lane_evidence

PIPELINE_VERSION = "v4-hybrid-2026-10-04"
MODEL = "models/fastseg_large_512x1024.onnx"
WIDTH = 1280
ROI_TOP = 0.62          # road-gate region starts this far down the image
ROI_BOTTOM = 0.90       # default bottom cut (raised further if bonnet detected)
ROAD_MIN_FRAC = 0.10    # min surface share of the road-gate region
SEG_H, SEG_W = 512, 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ROAD, SIDEWALK = 0, 1
VEHICLES = [11, 12, 13, 14, 15, 16, 17, 18]   # person, rider, car, truck, bus, train, motorcycle, bicycle
STATUSES = ["painted", "inferred", "no_lanes", "not_road", "review", "unusable"]


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


def _result(status, **kw):
    r = dict(status=status, reason="", lanes_detected=0, lanes_painted=0, lane_conf="",
             road_frac="", drivable_frac="", surface_frac="", ego_top="",
             _lanes=None, _painted=None, _poly=None, _vehicle=None)
    r.update(kw)
    return r


def to_record(r):
    """JSON/CSV-safe copy of a result (drops image masks)."""
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _load_net(model_path, engine):
    if engine == "classic":
        return cv.dnn.readNetFromONNX(model_path, engine=cv.dnn.ENGINE_CLASSIC)
    if engine == "new":
        return cv.dnn.readNetFromONNX(model_path, engine=cv.dnn.ENGINE_NEW)
    return cv.dnn.readNetFromONNX(model_path)


class RoadAuditor:
    def __init__(self, model_path=MODEL, engine=None, lane_model_path=LANE_MODEL_PATH):
        self.engine = (engine or os.environ.get("DNN_ENGINE", "classic")).lower()
        self.net = _load_net(model_path, self.engine)
        self.lanenet = LaneNet(lane_model_path)

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
        common = dict(_vehicle=vehicle, ego_top=round(ego_top / h, 2))
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

        lane_labels = labels.copy()
        lane_labels[ego_top:, :] = 255                     # own bonnet is never road
        lanes, conf = self.lanenet.detect(img)
        ev = lane_evidence(img, lane_labels, lanes)
        n_det = sum(e["detected"] for e in ev)
        n_paint = sum(e["is_painted"] for e in ev)
        status = "painted" if n_paint >= 1 else "inferred" if n_det >= 1 else "no_lanes"
        return _result(status, lanes_detected=n_det, lanes_painted=n_paint,
                       lane_conf=round(conf, 2), _lanes=lanes,
                       _painted=[e["is_painted"] for e in ev], **common)

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
        if r["_poly"] is not None:
            cv.polylines(out, r["_poly"], True, (255, 200, 0), 2)
        if r["_lanes"] is not None:
            for i, pts in enumerate(r["_lanes"]):
                if i in (1, 2):
                    colour = (0, 220, 0) if r["_painted"][i - 1] else (0, 165, 255)
                    radius = 7
                else:
                    colour, radius = (180, 180, 180), 4
                for p in pts:
                    cv.circle(out, p, radius, colour, -1)
        colour = {"painted": (0, 200, 0), "inferred": (0, 165, 255), "no_lanes": (0, 0, 255),
                  "not_road": (150, 150, 150), "review": (255, 255, 0),
                  "unusable": (100, 100, 100)}[r["status"]]
        label = r["status"].upper() + (f" ({r['reason']})" if r["reason"] else "")
        cv.putText(out, f"{label}  painted {r['lanes_painted']}/2  detected {r['lanes_detected']}/2",
                   (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
        return out
