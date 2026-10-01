import csv
import random
from collections import defaultdict

random.seed(7)
test_ids = {r["image_id"] for r in csv.DictReader(open("labels.csv"))}
by_area = defaultdict(list)
for r in csv.DictReader(open("output/results.csv")):
    if r["status"] != "unusable" and r["image_id"] not in test_ids:
        by_area[r["area"]].append(r)

sample = []
for area, rows in by_area.items():
    sample += random.sample(rows, min(15, len(rows)))

with open("dev_labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["area", "image_id", "image_path", "my_label"])
    for r in sample:
        w.writerow([r["area"], r["image_id"], f"data/{r['area']}/{r['image_id']}.jpg", ""])
print(f"Wrote dev_labels.csv with {len(sample)} images")
