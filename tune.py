"""Show the trade-off between catching bad roads and false alarms, on the DEV set only.
Usage: python tune.py <results.csv>
Binary view: 'needs attention' = degraded or fail.  READY_T controls this trade-off;
FAIL_T only splits degraded vs fail, so it is chosen to maximise exact agreement."""
import csv
import sys

results_file = sys.argv[1] if len(sys.argv) > 1 else "output_v3c/results.csv"
preds = {r["image_id"]: r for r in csv.DictReader(open(results_file))}
data = []
for r in csv.DictReader(open("dev_labels.csv")):
    p = preds.get(r["image_id"])
    if p and r["my_label"] in ("ready", "degraded", "fail") and p["status"] in ("ready", "degraded", "fail"):
        data.append((r["my_label"], float(p["paint_contrast"])))

n_bad = sum(h in ("degraded", "fail") for h, _ in data)
n_good = sum(h == "ready" for h, _ in data)


def classify(c, ready_t, fail_t):
    if c >= ready_t:
        return "ready"
    return "degraded" if c >= fail_t else "fail"


print(f"{results_file}: {len(data)} DEV images graded ({n_bad} bad, {n_good} good)\n")
print(f"{'READY_T':>8s}{'best FAIL_T':>12s}{'bad caught':>12s}{'false alarms':>14s}{'binary acc':>12s}{'exact':>8s}")
for ready_t in range(40, 125, 5):
    best = None
    for fail_t in range(20, ready_t + 1, 5):
        preds_ = [(h, classify(c, ready_t, fail_t)) for h, c in data]
        exact = sum(h == p for h, p in preds_)
        if best is None or exact > best[1]:
            best = (fail_t, exact, preds_)
    fail_t, exact, preds_ = best
    caught = sum(h != "ready" and p != "ready" for h, p in preds_)
    false_alarms = sum(h == "ready" and p != "ready" for h, p in preds_)
    binary = sum((h == "ready") == (p == "ready") for h, p in preds_)
    print(f"{ready_t:>8d}{fail_t:>12d}{f'{caught}/{n_bad}':>12s}{f'{false_alarms}/{n_good}':>14s}"
          f"{f'{binary}/{len(data)}':>12s}{f'{exact}/{len(data)}':>8s}")
