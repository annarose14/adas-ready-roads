"""Usage: python evaluate.py <labels.csv> <results.csv>"""
import csv
import sys
from collections import Counter
from pathlib import Path

labels_file, results_file = sys.argv[1], sys.argv[2]
LEVELS = ["ready", "degraded", "fail"]
COLS = ["ready", "degraded", "fail", "no_lanes", "not_road", "review"]

preds = {r["image_id"]: r["status"] for r in csv.DictReader(open(results_file))}
pairs = [(r["my_label"], preds.get(r["image_id"], "missing"))
         for r in csv.DictReader(open(labels_file)) if r["my_label"]]

lines = []
def out(s=""):
    print(s)
    lines.append(s)

out(f"Labels: {labels_file}   Results: {results_file}")
out(f"Images: {len(pairs)}  |  Your labels: " +
    ", ".join(f"{k}={v}" for k, v in Counter(h for h, _ in pairs).items()))
out()
cm = Counter(pairs)
out("Confusion matrix (rows = YOU, columns = DETECTOR)")
out(f"{'':12s}" + "".join(f"{c:>10s}" for c in COLS))
for r in ["ready", "degraded", "fail", "none", "unusable"]:
    if any(cm[(r, c)] for c in COLS):
        out(f"{r:12s}" + "".join(f"{cm[(r, c)]:>10d}" for c in COLS))
out()

graded = [(h, p) for h, p in pairs if h in LEVELS]
if graded:
    exact = sum(h == p for h, p in graded)
    always_ready = sum(h == "ready" for h, _ in graded)
    out(f"Lane-lines-expected images: {len(graded)}")
    out(f"  Exact agreement:         {exact}/{len(graded)} = {100*exact/len(graded):.0f}%")
    out(f"  'Always ready' baseline: {always_ready}/{len(graded)} = {100*always_ready/len(graded):.0f}%")
    bad = [(h, p) for h, p in graded if h in ("degraded", "fail")]
    if bad:
        missed = sum(p == "ready" for _, p in bad)
        out(f"  Bad roads wrongly called READY: {missed}/{len(bad)}  <- most dangerous error")
    n_ready = sum(h == "ready" for h, _ in graded)
    good_lost = sum(h == "ready" and p in ("no_lanes", "not_road") for h, p in graded)
    out(f"  Good roads wrongly skipped (no_lanes/not_road): {good_lost}/{n_ready}")
    decided = [(h, p) for h, p in graded if p in LEVELS]
    if decided:
        d_exact = sum(h == p for h, p in decided)
        out(f"  Agreement when detector gave a grade: {d_exact}/{len(decided)} = {100*d_exact/len(decided):.0f}%")

review = sum(p == "review" for _, p in pairs)
out(f"Sent to review (human/agent check): {review}/{len(pairs)}")

none_preds = [p for h, p in pairs if h == "none"]
if none_preds:
    correct = sum(p in ("no_lanes", "not_road") for p in none_preds)
    out(f"No-markings-expected images: {len(none_preds)} -> correctly skipped {correct}, detector said " +
        ", ".join(f"{k}={v}" for k, v in Counter(none_preds).items()))

name = f"eval_{Path(labels_file).stem}_{Path(results_file).parent.name}.txt"
Path(name).write_text("\n".join(lines))
print(f"\nSaved {name}")
