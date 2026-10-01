"""Grid-search contrast thresholds on the DEV set only. Never run this on test_labels.csv."""
import csv

preds = {r["image_id"]: r for r in csv.DictReader(open("output_v1/results.csv"))}
data = []
for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if p and r["my_label"] in ("ready", "degraded", "fail") and p["status"] != "unusable":
        contrast = float(p["paint_contrast"]) if p["paint_contrast"] else 0.0
        data.append((r["my_label"], contrast))

def classify(c, ready_t, fail_t):
    if c >= ready_t:
        return "ready"
    if c >= fail_t:
        return "degraded"
    return "fail"

results = []
for ready_t in range(50, 121, 5):
    for fail_t in range(30, ready_t, 5):
        preds_ = [(h, classify(c, ready_t, fail_t)) for h, c in data]
        exact = sum(h == p for h, p in preds_)
        bad = [(h, p) for h, p in preds_ if h in ("degraded", "fail")]
        missed = sum(p == "ready" for _, p in bad)
        good_rejected = sum(h == "ready" and p == "fail" for h, p in preds_)
        results.append((missed, -exact, good_rejected, ready_t, fail_t, exact, len(bad)))

results.sort()
n = len(data)
print(f"Dev images with lane lines expected: {n}")
print(f"{'ready_t':>8s}{'fail_t':>8s}{'dangerous misses':>18s}{'exact':>10s}{'good->fail':>12s}")
for missed, _, good_rej, rt, ft, exact, nbad in results[:15]:
    print(f"{rt:>8d}{ft:>8d}{f'{missed}/{nbad}':>18s}{f'{exact}/{n}':>10s}{good_rej:>12d}")
