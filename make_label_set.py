import csv
import random
from collections import defaultdict

random.seed(42)
by_area = defaultdict(list)
with open("output/results.csv") as f:
    for row in csv.DictReader(f):
        if row["status"] != "unusable":
            by_area[row["area"]].append(row)

sample = []
for area, rows in by_area.items():
    sample += random.sample(rows, min(15, len(rows)))

with open("labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["area", "image_id", "image_path", "my_label"])
    for r in sample:
        w.writerow([r["area"], r["image_id"], f"data/{r['area']}/{r['image_id']}.jpg", ""])
print(f"Wrote labels.csv with {len(sample)} images to label")
