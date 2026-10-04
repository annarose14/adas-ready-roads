"""Carry location labels over to a NEW run whose location IDs differ, by matching shared photos.
Usage: python remap_labels.py <old_labels.csv> <old_agent_frames.csv> <new_agent_frames.csv> <out_labels.csv>
Each new location gets the label of the labelled old location(s) it shares the most original
photos with (overlap-weighted majority). New locations sharing no labelled photos are skipped."""
import csv
import sys
from collections import Counter, defaultdict

old_labels_f, old_frames_f, new_frames_f, out_f = sys.argv[1:5]
labels = {r["location_id"]: r["label"] for r in csv.DictReader(open(old_labels_f))}

old_by_img = defaultdict(set)
for r in csv.DictReader(open(old_frames_f)):
    if r["segment_id"] in labels and r["origin"] == "original":
        old_by_img[r["image_id"]].add(r["segment_id"])

new_imgs = defaultdict(set)
for r in csv.DictReader(open(new_frames_f)):
    if r["origin"] == "original":
        new_imgs[r["segment_id"]].add(r["image_id"])

rows, mixed, skipped = [], 0, 0
for loc, imgs in sorted(new_imgs.items()):
    votes = Counter()
    for img in imgs:
        for old in old_by_img.get(img, ()):
            votes[labels[old]] += 1
    if not votes:
        skipped += 1
        continue
    label, n = votes.most_common(1)[0]
    if len(votes) > 1:
        mixed += 1
    rows.append((loc, label, n, sum(votes.values())))

with open(out_f, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["location_id", "label"])
    for loc, label, _, _ in rows:
        w.writerow([loc, label])
print(f"Mapped {len(rows)} new locations to labels ({mixed} had mixed old labels -> majority used); "
      f"{skipped} new locations had no labelled photos")
