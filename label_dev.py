"""Keyboard labelling tool: shows each image, press a key, saves to labels.csv."""
import csv
import cv2 as cv

KEYS = {ord("r"): "ready", ord("d"): "degraded", ord("f"): "fail",
        ord("n"): "none", ord("x"): "unusable"}

with open("dev_labels.csv") as f:
    rows = list(csv.DictReader(f))
fields = ["area", "image_id", "image_path", "my_label"]


def save():
    with open("dev_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


todo = [i for i, r in enumerate(rows) if not r["my_label"]]
print(f"{len(todo)} images left to label")

for n, i in enumerate(todo, 1):
    img = cv.imread(rows[i]["image_path"])
    if img is None:
        continue
    h, w = img.shape[:2]
    img = cv.resize(img, (1280, int(h * 1280 / w)))
    cv.rectangle(img, (0, 0), (1280, 45), (0, 0, 0), -1)
    cv.putText(img, f"{n}/{len(todo)}   r=ready  d=degraded  f=fail  n=none  x=bad image  q=quit",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.imshow("label", img)
    while True:
        key = cv.waitKey(0) & 0xFF
        if key == ord("q"):
            save()
            cv.destroyAllWindows()
            raise SystemExit("Saved. Run again to continue where you left off.")
        if key in KEYS:
            rows[i]["my_label"] = KEYS[key]
            save()
            break

cv.destroyAllWindows()
print("All done - labels.csv saved.")
