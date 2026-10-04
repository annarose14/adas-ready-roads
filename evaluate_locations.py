"""Evaluate the location agent against blind human location labels.
Usage: python evaluate_locations.py [labels_csv] [system_csv]
       defaults: location_labels_test.csv test_segments_agent.csv
Output file: eval_<system_csv stem>.txt  (never overwrites results for other systems)"""
import csv
import sys
from collections import Counter
from pathlib import Path

labels_file = sys.argv[1] if len(sys.argv) > 1 else "location_labels_test.csv"
system_file = sys.argv[2] if len(sys.argv) > 2 else "test_segments_agent.csv"
SYS = {"adas_readable": "readable", "at_risk": "at_risk", "not_readable": "not_readable",
       "unmarked_by_design": "unmarked", "human_review": "review"}
HUMAN = ["readable", "at_risk", "not_readable", "unmarked", "cant_tell"]
COLS = ["readable", "at_risk", "not_readable", "unmarked", "review"]

human = {r["location_id"]: r["label"] for r in csv.DictReader(open(labels_file))}
system = {r["segment_id"]: SYS[r["final_verdict"]] for r in csv.DictReader(open(system_file))}
pairs = [(human[k], system[k]) for k in sorted(human) if k in system]

lines = []
def out(s=""):
    print(s)
    lines.append(s)

out(f"Location-level evaluation: {system_file} vs {labels_file}  ({len(pairs)} locations)")
out("Human labels: " + ", ".join(f"{k}={v}" for k, v in Counter(h for h, _ in pairs).items()))
out("System:       " + ", ".join(f"{k}={v}" for k, v in Counter(s for _, s in pairs).items()))
out()
cm = Counter(pairs)
out("Confusion matrix (rows = HUMAN, columns = SYSTEM)")
out(f"{'':14s}" + "".join(f"{c:>13s}" for c in COLS))
for h in HUMAN:
    if any(cm[(h, c)] for c in COLS):
        out(f"{h:14s}" + "".join(f"{cm[(h, c)]:>13d}" for c in COLS))
out()

decided = [(h, s) for h, s in pairs if s != "review"]
out(f"Coverage: system decided {len(decided)}/{len(pairs)} ({100 * len(decided) / len(pairs):.0f}%)")
judgeable = [(h, s) for h, s in decided if h != "cant_tell"]
if judgeable:
    exact = sum(h == s for h, s in judgeable)
    common = Counter(h for h, _ in judgeable).most_common(1)[0]
    out(f"Exact agreement (4 classes): {exact}/{len(judgeable)} = {100 * exact / len(judgeable):.0f}%"
        f"   (baseline 'always {common[0]}': {100 * common[1] / len(judgeable):.0f}%)")
    marked = lambda x: x in ("readable", "at_risk", "not_readable")
    mk = sum(marked(h) == marked(s) for h, s in judgeable)
    out(f"MARKED vs UNMARKED agreement:  {mk}/{len(judgeable)} = {100 * mk / len(judgeable):.0f}%")
bad = [(h, s) for h, s in judgeable if h in ("at_risk", "not_readable")]
if bad:
    out(f"Problem roads decided: {len(bad)} -> caught {sum(s in ('at_risk', 'not_readable') for _, s in bad)}, "
        f"DANGEROUS (called readable) {sum(s == 'readable' for _, s in bad)}")
good = [(h, s) for h, s in judgeable if h == "readable"]
if good:
    out(f"Readable roads: {len(good)} -> false alarms {sum(s in ('at_risk', 'not_readable') for _, s in good)}")
um = [(h, s) for h, s in pairs if h == "unmarked"]
if um:
    out(f"Unmarked roads: {len(um)} -> system said unmarked {sum(s == 'unmarked' for _, s in um)}")
rev = [h for h, s in pairs if s == "review"]
if rev:
    out("Sent to review -> human said: " + ", ".join(f"{k}={v}" for k, v in Counter(rev).items()))

name = f"eval_{Path(system_file).stem}.txt"
Path(name).write_text("\n".join(lines))
print(f"\nSaved {name}")
