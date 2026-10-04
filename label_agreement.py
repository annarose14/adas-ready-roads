"""Intra-rater reliability: compare two blind labelling passes of the same locations.
Usage: python label_agreement.py <labels_a.csv> <labels_b.csv>
Reports raw agreement and Cohen's kappa overall, and on the two key distinctions."""
import csv
import sys
from collections import Counter

a = {r["location_id"]: r["label"] for r in csv.DictReader(open(sys.argv[1]))}
b = {r["location_id"]: r["label"] for r in csv.DictReader(open(sys.argv[2]))}
keys = sorted(set(a) & set(b))
pairs = [(a[k], b[k]) for k in keys]


def kappa(ps):
    n = len(ps)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in ps) / n
    ca, cb = Counter(x for x, _ in ps), Counter(y for _, y in ps)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report(name, ps):
    n = len(ps)
    agree = sum(x == y for x, y in ps)
    print(f"{name:42s} n={n:3d}  agreement {agree}/{n} = {100 * agree / max(n, 1):.0f}%  kappa {kappa(ps):.2f}")


print(f"Pass A: {sys.argv[1]}   Pass B: {sys.argv[2]}   locations in both: {len(keys)}\n")
labels = sorted(set(x for p in pairs for x in p))
print("Confusion (rows = pass A, columns = pass B)")
print(f"{'':14s}" + "".join(f"{l:>13s}" for l in labels))
cm = Counter(pairs)
for r in labels:
    print(f"{r:14s}" + "".join(f"{cm[(r, c)]:>13d}" for c in labels))
print()
report("All labels", pairs)
marked = lambda x: "marked" if x in ("readable", "at_risk", "not_readable") else x
report("Marked vs unmarked (vs can't tell)", [(marked(x), marked(y)) for x, y in pairs])
rv = [(x, y) for x, y in pairs if x in ("readable", "at_risk", "not_readable")
      and y in ("readable", "at_risk", "not_readable")]
report("Readable vs at-risk vs not (marked roads only)", rv)
print("\nKappa guide: <0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 near-perfect")
