"""Step 7a (development): compare the agent's evidence on locations a human labelled
yes vs no. Diagnosis only; nothing here changes the agent."""
import csv
import statistics
import sys

labels_path, segs_path = sys.argv[1], sys.argv[2]
lab_rows = list(csv.DictReader(open(labels_path)))
key = list(lab_rows[0].keys())[0]
labels = {r[key]: r["label"] for r in lab_rows}
segs = {r["segment_id"]: r for r in csv.DictReader(open(segs_path))}
print("label key:", key, "| segment columns:", ", ".join(list(next(iter(segs.values())).keys())[:40]))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


print("\nHUMAN  FINAL_VERDICT        VLM yes/no/asked  OPENCV_PAINTED  HIGHWAY       ID")
groups = {"yes": [], "no": []}
for lid, lab in sorted(labels.items(), key=lambda x: x[1]):
    s = segs.get(lid)
    if not s or lab not in groups:
        continue
    groups[lab].append(s)
    print(f"{lab:6} {s.get('final_verdict', ''):20} {s.get('vlm_yes', '')}/{s.get('vlm_no', '')}/{s.get('vlm_asked', ''):12} "
          f"{s.get('opencv_painted_rate', ''):15} {s.get('highway', ''):13} {lid}")

for lab, rows in groups.items():
    pr = [num(s.get("opencv_painted_rate")) for s in rows]
    pr = [v for v in pr if v is not None]
    if pr:
        print(f"\nhuman {lab}: n={len(rows)}  OpenCV painted rate median {statistics.median(pr):.2f}, "
              f"share at 0.00: {sum(v == 0 for v in pr)}/{len(pr)}")
