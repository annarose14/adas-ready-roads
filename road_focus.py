"""ADAS-Ready Roads - OpenCV 5 road focus for the vision language model.
road_view(img, auditor, mode) uses the fastseg segmentation (OpenCV DNN) to find the road
surface ahead, removes the camera car's own bonnet, and returns
  mode "crop": the photo cropped to the road region
  mode "mask": the same crop with everything that is not road painted flat grey (road kept with
               a small margin so paint at the kerb survives)
It returns None when there is too little road to show.
ask_view(path, image_id, mode, auditor) asks the VLM about that view (mode "full" = whole photo).
Answers are cached in vlm_cache.json under '<model>|<mode>|<image id>'."""
import json

import cv2 as cv
import numpy as np

import vlm_tool
from adas_pipeline import ROAD, VEHICLES, _find_ego_top, resize

TOP_FRAC = 0.35          # ignore rows above this share of the height (sky, far distance)
MIN_ROAD_FRAC = 0.02     # less road than this share of the photo -> no view
MARGIN_PX = 15           # keep pixels this close to the road so kerb side paint survives
PAD_FRAC = 0.03          # padding around the road box, as a share of the width
GREY = 128


def road_view(img, auditor, mode="crop"):
    img = resize(img)
    h, w = img.shape[:2]
    labels = auditor.segment(img)
    vehicle = np.where(np.isin(labels, VEHICLES), 255, 0).astype(np.uint8)
    ego_top = _find_ego_top(vehicle, h, w)
    road = np.where(labels == ROAD, 255, 0).astype(np.uint8)
    road[ego_top:, :] = 0
    y_lim = int(TOP_FRAC * h)
    road[:y_lim, :] = 0
    if cv.countNonZero(road) < MIN_ROAD_FRAC * h * w:
        return None
    ys, xs = np.nonzero(road)
    pad = int(PAD_FRAC * w)
    y0 = max(y_lim, int(np.percentile(ys, 1)) - pad)
    y1 = min(ego_top, int(np.percentile(ys, 99.5)) + pad)
    x0 = max(0, int(np.percentile(xs, 1)) - pad)
    x1 = min(w, int(np.percentile(xs, 99)) + pad)
    if y1 - y0 < 100 or x1 - x0 < 200:
        return None
    view = img[y0:y1, x0:x1].copy()
    if mode == "mask":
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * MARGIN_PX + 1, 2 * MARGIN_PX + 1))
        keep = cv.dilate(road, k)[y0:y1, x0:x1]
        view[keep == 0] = GREY
    return view


def ask_view(path, image_id, mode, auditor):
    if mode == "full":
        return vlm_tool.ask_lines_visible(path, image_id)
    cache = vlm_tool._get_cache()
    key = f"{vlm_tool.MODEL_ID}|{mode}|{image_id}"
    if key in cache:
        return cache[key]
    img = cv.imread(str(path))
    if img is None:
        return "UNCLEAR"
    view = road_view(img, auditor, mode)
    answer = "NO_ROAD" if view is None else vlm_tool.ask_image(view)
    cache[key] = answer
    vlm_tool.CACHE.write_text(json.dumps(cache))
    return answer
