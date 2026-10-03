"""One-time evaluation of the location agent against blind human labels (fresh TEST suburbs).
Usage: python evaluate_locations.py [labels_csv] [system_csv]
       defaults: location_labels_test.csv test_segments_agent.csv"""
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

out(f"Location-level TEST evaluation  ({len(pairs)} locations, fresh suburbs, blind labels)")
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
out(f"Coverage: system decided {len(decided)}/{len(pairs)} locations "
    f"({100 * len(decided) / len(pairs):.0f}%), sent {len(pairs) - len(decided)} to human review")

judgeable = [(h, s) for h, s in decided if h != "cant_tell"]
if judgeable:
    exact = sum(h == s for h, s in judgeable)
    common = Counter(h for h, _ in judgeable).most_common(1)[0]
    out(f"Agreement on decided + human-judgeable: {exact}/{len(judgeable)} = {100 * exact / len(judgeable):.0f}%"
        f"   (baseline 'always {common[0]}': {common[1]}/{len(judgeable)} = {100 * common[1] / len(judgeable):.0f}%)")

bad = [(h, s) for h, s in judgeable if h in ("at_risk", "not_readable")]
if bad:
    missed = sum(s == "readable" for _, s in bad)
    caught = sum(s in ("at_risk", "not_readable") for _, s in bad)
    out(f"Problem roads (human at_risk/not_readable) that the system decided on: {len(bad)}")
    out(f"  caught as at_risk/not_readable: {caught}   DANGEROUS (called readable): {missed}")
good = [(h, s) for h, s in judgeable if h == "readable"]
if good:
    fa = sum(s in ("at_risk", "not_readable") for _, s in good)
    out(f"Readable roads (human): {len(good)}   false alarms (flagged at_risk/not_readable): {fa}")
um = [(h, s) for h, s in pairs if h == "unmarked"]
if um:
    out(f"Unmarked-by-design roads (human): {len(um)}   system agreed: {sum(s == 'unmarked' for _, s in um)}")
rev = [h for h, s in pairs if s == "review"]
if rev:
    out("Sent to review -> human said: " + ", ".join(f"{k}={v}" for k, v in Counter(rev).items()))

Path("eval_locations_test.txt").write_text("\n".join(lines))
print("\nSaved eval_locations_test.txt")
