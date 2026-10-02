"""How many usable photos have neighbours (same sequence, within 50 m)?
Decides whether segment-level consensus is feasible."""
import csv
import math
from collections import Counter, defaultdict


def dist_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 6371000 * 2 * math.asin(math.sqrt(a))


rows = [r for r in csv.DictReader(open("output_final/results.csv"))
        if r["status"] != "unusable" and r["lat"] and r["lon"]]
by_seq = defaultdict(list)
for r in rows:
    by_seq[r["sequence"]].append(r)

neigh_counts = []
for r in rows:
    lat, lon = float(r["lat"]), float(r["lon"])
    n = sum(1 for o in by_seq[r["sequence"]] if o is not r and
            dist_m(lat, lon, float(o["lat"]), float(o["lon"])) <= 50)
    neigh_counts.append(n)

print(f"Usable photos with GPS: {len(rows)}   Sequences: {len(by_seq)}")
print(f"Photos per sequence: " + ", ".join(f"{len(v)}" for v in sorted(by_seq.values(), key=len, reverse=True)[:15]) + " ...")
c = Counter(min(n, 5) for n in neigh_counts)
print("Neighbours within 50 m (same sequence):")
for k in range(6):
    label = f"{k}+" if k == 5 else str(k)
    print(f"  {label:>3s} neighbours: {c.get(k, 0)} photos")
