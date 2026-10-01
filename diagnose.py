"""Show v1 measurements grouped by YOUR label, to see if they separate good vs bad roads."""
import csv
from collections import defaultdict
import statistics as st

preds = {r["image_id"]: r for r in csv.DictReader(open("output_v1/results.csv"))}
groups = defaultdict(list)
for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if p and p["status"] not in ("unusable",) and r["my_label"] in ("ready", "degraded", "fail"):
        groups[r["my_label"]].append(p)

def num(rows, key):
    return [float(p[key]) for p in rows if p[key] not in ("", None)]

print(f"{'YOUR LABEL':12s}{'n':>4s}   {'score (median)':>15s}{'continuity':>12s}{'contrast':>10s}")
for label in ("ready", "degraded", "fail"):
    rows = groups[label]
    if not rows:
        continue
    s, c, k = num(rows, "score"), num(rows, "continuity"), num(rows, "paint_contrast")
    med = lambda v: f"{st.median(v):.2f}" if v else "-"
    print(f"{label:12s}{len(rows):>4d}   {med(s):>15s}{med(c):>12s}{med(k):>10s}")

print("\nEvery dev image (sorted by score):")
print(f"{'YOU':10s}{'DETECTOR':10s}{'score':>7s}{'cont':>7s}{'contr':>7s}  image")
allrows = [(lbl, p) for lbl, rows in groups.items() for p in rows]
for lbl, p in sorted(allrows, key=lambda x: float(x[1]["score"] or 0)):
    print(f"{lbl:10s}{p['status']:10s}{p['score']:>7s}{p['continuity']:>7s}{p['paint_contrast']:>7s}  {p['area']}/{p['image_id']}.jpg")
