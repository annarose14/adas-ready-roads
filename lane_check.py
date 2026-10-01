"""
ADAS-Ready Roads - v0 baseline lane-marking visibility checker (OpenCV 5).
Reads data/<area>/*.jpg, writes annotated images to output/<area>/ and output/results.csv
"""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

DATA = Path("data")
OUT = Path("output")
WIDTH = 1280


def resize(img):
    h, w = img.shape[:2]
    scale = WIDTH / w
    return cv.resize(img, (WIDTH, int(h * scale)))


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
    poly = np.array([[(0, h), (w, h),
                      (int(w * 0.6), int(h * 0.55)),
                      (int(w * 0.4), int(h * 0.55))]], np.int32)
    cv.fillPoly(mask, poly, 255)
    return mask, poly


def analyse(img):
    h, w = img.shape[:2]
    roi, poly = road_roi(h, w)
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    lightness = hls[:, :, 1]

    white = cv.inRange(hls, (0, 190, 0), (180, 255, 80))
    yellow = cv.inRange(hls, (15, 80, 90), (35, 220, 255))
    marks = cv.bitwise_and(cv.bitwise_or(white, yellow), roi)

    edges = cv.Canny(cv.GaussianBlur(marks, (5, 5), 0), 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                           minLineLength=40, maxLineGap=20)
    kept = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if 0.3 < abs(slope) < 3:
                kept.append((int(x1), int(y1), int(x2), int(y2)))

    total_len = sum(np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in kept)
    road = cv.bitwise_and(roi, cv.bitwise_not(marks))
    mark_l = cv.mean(lightness, mask=marks)[0] if cv.countNonZero(marks) > 50 else 0
    road_l = cv.mean(lightness, mask=road)[0]
    contrast = max(0.0, mark_l - road_l)

    len_score = min(total_len / (h * 0.8), 1.0)
    contrast_score = min(contrast / 80, 1.0)
    score = round(100 * (0.6 * len_score + 0.4 * contrast_score), 1)

    if score >= 60:
        status = "ready"
    elif score >= 30:
        status = "degraded"
    else:
        status = "fail"
    return score, status, kept, poly, round(contrast, 1), round(total_len)


def annotate(img, score, status, lines, poly):
    out = img.copy()
    cv.polylines(out, poly, True, (255, 200, 0), 2)
    for x1, y1, x2, y2 in lines:
        cv.line(out, (x1, y1), (x2, y2), (0, 255, 0), 4)
    colour = {"ready": (0, 200, 0), "degraded": (0, 200, 255), "fail": (0, 0, 255)}[status]
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
            status, score, contrast, length = "unusable", "", "", ""
        else:
            score, status, lines, poly, contrast, length = analyse(img)
            (OUT / area).mkdir(exist_ok=True)
            cv.imwrite(str(OUT / area / path.name), annotate(img, score, status, lines, poly))
        counts[status] = counts.get(status, 0) + 1
        rows.append([area, path.stem, q, score, status, contrast, length])

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "image_id", "quality", "score", "status", "contrast", "line_length_px"])
        w.writerows(rows)

    print(f"OpenCV {cv.__version__} - processed {len(rows)} images")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s} {v}")
    print("Annotated images in output/<area>/, table in output/results.csv")


if __name__ == "__main__":
    main()
