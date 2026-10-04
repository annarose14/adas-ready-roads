"""Evaluate a location agent run against blind human labels on "Are painted lane lines visible?"
System mapping: adas_readable/at_risk -> yes, not_readable/unmarked_by_design -> no,
human_review -> abstains (routed to a person).
Metrics (fixed BEFORE the clean Penrith/Sutherland test): agreement + Wilson 95% CI, Cohen's kappa,
per-class recall, BALANCED accuracy (mean of yes-recall and no-recall; robust to class imbalance).
Baselines: majority answer (balanced accuracy 50% by definition); OpenStreetMap-only.
Usage: python evaluate_binary.py <labels_csv> <system_csv>     Output: eval_<system stem>.txt"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

labels_f = sys.argv[1] if len(sys.argv) > 1 else "labels_final.csv"
system_f = sys.argv[2] if len(sys.argv) > 2 else "final_segments_agent.csv"
SYS = {"adas_readable": "yes", "at_risk": "yes", "not_readable": "no",
       "unmarked_by_design": "no", "human_review": "review"}

human = {r["location_id"]: r["label"] for r in csv.DictReader(open(labels_f))}
rows = {r["segment_id"]: r for r in csv.DictReader(open(system_f))}
keys = [k for k in sorted(human) if k in rows]

lines = []
def out(s=""):
    print(s)
    lines.append(s)


def kappa(ps):
    n = len(ps)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in ps) / n
    ca, cb = Counter(a for a, _ in ps), Counter(b for _, b in ps)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}%  [95% CI {lo:.0f}-{hi:.0f}%]"


def balanced(ps):
    rec = []
    for c in ("yes", "no"):
        sub = [(a, b) for a, b in ps if a == c]
        if sub:
            rec.append(sum(a == b for a, b in sub) / len(sub))
    return 100 * sum(rec) / len(rec) if rec else float("nan")


sys_pred = {k: SYS[rows[k]["final_verdict"]] for k in keys}
osm_pred = {k: {"True": "yes", "False": "no"}.get(rows[k]["expected_marked"], "unknown") for k in keys}

out(f"EVALUATION: {system_f} vs {labels_f}  ({len(keys)} locations)")
out("Question: are painted lane lines visible?")
out("Human:  " + ", ".join(f"{k}={v}" for k, v in Counter(human[k] for k in keys).items()))
out("System: " + ", ".join(f"{k}={v}" for k, v in Counter(sys_pred.values()).items()))
out()
cm = Counter((human[k], sys_pred[k]) for k in keys)
out("Confusion (rows = HUMAN, columns = SYSTEM)")
out(f"{'':12s}{'yes':>8s}{'no':>8s}{'review':>8s}")
for h in ("yes", "no", "cant_tell"):
    if any(cm[(h, c)] for c in ("yes", "no", "review")):
        out(f"{h:12s}{cm[(h, 'yes')]:>8d}{cm[(h, 'no')]:>8d}{cm[(h, 'review')]:>8d}")
out()

judge = [k for k in keys if human[k] in ("yes", "no")]
decided = [k for k in judge if sys_pred[k] in ("yes", "no")]
out(f"Coverage: system decided {sum(sys_pred[k] != 'review' for k in keys)}/{len(keys)}; "
    f"human could judge {len(judge)}/{len(keys)}")
if decided:
    ps = [(human[k], sys_pred[k]) for k in decided]
    agree = sum(a == b for a, b in ps)
    out(f"SYSTEM agreement (decided & judgeable): {pct(agree, len(ps))}   kappa {kappa(ps):.2f}")
    out(f"SYSTEM balanced accuracy: {balanced(ps):.0f}%   (majority baseline = 50% by definition)")
    yes_h = [k for k in decided if human[k] == "yes"]
    no_h = [k for k in decided if human[k] == "no"]
    if yes_h:
        out(f"  lines visible (human yes): system found them  {pct(sum(sys_pred[k] == 'yes' for k in yes_h), len(yes_h))}")
    if no_h:
        out(f"  no lines (human no):       system said none   {pct(sum(sys_pred[k] == 'no' for k in no_h), len(no_h))}")
if judge:
    maj, n_maj = Counter(human[k] for k in judge).most_common(1)[0]
    out(f"Baseline 'always {maj}': {pct(n_maj, len(judge))}   balanced accuracy 50%")
    osm = [k for k in judge if osm_pred[k] in ("yes", "no")]
    if osm:
        ps = [(human[k], osm_pred[k]) for k in osm]
        agree = sum(a == b for a, b in ps)
        out(f"Baseline OpenStreetMap-only: {pct(agree, len(ps))}   kappa {kappa(ps):.2f}   "
            f"balanced accuracy {balanced(ps):.0f}%   ({len(judge) - len(osm)} with unknown OSM type)")
rev = [human[k] for k in keys if sys_pred[k] == "review"]
if rev:
    out("Sent to human review -> human said: " + ", ".join(f"{k}={v}" for k, v in Counter(rev).items()))

name = f"eval_{Path(system_f).stem}.txt"
Path(name).write_text("\n".join(lines))
print(f"\nSaved {name}")
