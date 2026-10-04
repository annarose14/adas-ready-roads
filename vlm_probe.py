"""Feasibility probe (DEV data only): can a vision-language model on Amazon Bedrock answer
"are painted lane lines visible?" the way a human does?
Ground truth = former test suburbs, ONLY locations where BOTH blind labelling passes agreed on
marked (readable/at_risk -> yes) vs unmarked (-> no). Up to 4 photos per location, majority vote.
Answers are cached in vlm_cache.json, so re-runs resume where they stopped (e.g. if AWS blocks
access while the account is being verified).
Usage: python vlm_probe.py <bedrock_model_id>"""
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import boto3
import cv2 as cv
from botocore.exceptions import ClientError

MODEL_ID = sys.argv[1]
REGION = "ap-southeast-2"
PHOTOS_PER_LOCATION = 4
CACHE = Path("vlm_cache.json")
PROMPT = ("You are checking street-level photos for road-marking visibility. "
          "Look only at the road surface. Are painted lane lines (centre line, lane dividers, "
          "or painted edge lines) visible on the road in this photo? Ignore parking-bay lines, "
          "pedestrian crossings and kerbs. Answer with exactly one word: YES, NO, or UNCLEAR "
          "(UNCLEAR if the photo does not show a road from a vehicle's viewpoint).")

client = boto3.client("bedrock-runtime", region_name=REGION)
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


class Blocked(Exception):
    pass


def ask(path, image_id):
    key = f"{MODEL_ID}|{image_id}"
    if key in cache:
        return cache[key]
    img = cv.imread(str(path))
    h, w = img.shape[:2]
    if w > 1024:
        img = cv.resize(img, (1024, int(h * 1024 / w)))
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 85])
    for attempt in range(4):
        try:
            resp = client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": buf.tobytes()}}},
                    {"text": PROMPT}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0})
            text = resp["output"]["message"]["content"][0]["text"].strip().upper()
            answer = next((w for w in ("YES", "NO", "UNCLEAR") if text.startswith(w)), "UNCLEAR")
            cache[key] = answer
            CACHE.write_text(json.dumps(cache))
            return answer
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException":
                time.sleep(2 * (attempt + 1))
                continue
            if code == "AccessDeniedException":
                raise Blocked(e.response["Error"]["Message"][:120])
            raise
    return "UNCLEAR"


def label_map(path):
    return {r["location_id"]: r["label"] for r in csv.DictReader(open(path))}


a, b = label_map("location_labels_test.csv"), label_map("location_labels_test_relabel.csv")
marked = lambda x: "yes" if x in ("readable", "at_risk", "not_readable") else "no" if x == "unmarked" else None
truth = {k: marked(a[k]) for k in a if k in b and marked(a[k]) and marked(a[k]) == marked(b[k])}

frames = defaultdict(list)
for r in csv.DictReader(open("test_agent_frames.csv")):
    if r["segment_id"] in truth and r["origin"] == "original":
        frames[r["segment_id"]].append(r["image_id"])
roots = [p for base in Path(".").glob("data*") if base.is_dir() and base.name != "data_agent"
         for p in base.iterdir() if p.is_dir()]


def find(image_id):
    for d in roots:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


rng = random.Random(1)
results, blocked = [], None
for loc in sorted(truth):
    ids = frames.get(loc, [])
    rng.shuffle(ids)
    answers = []
    try:
        for image_id in ids[:PHOTOS_PER_LOCATION]:
            p = find(image_id)
            if p:
                answers.append(ask(p, image_id))
    except Blocked as e:
        blocked = str(e)
        break
    votes = Counter(x for x in answers if x != "UNCLEAR")
    pred = "yes" if votes["YES"] > votes["NO"] else "no" if votes["NO"] > votes["YES"] else "unclear"
    results.append((loc, truth[loc], pred))
    print(f"  {loc}  truth={truth[loc]:3s}  vlm={pred:7s}  photos={''.join(x[0] for x in answers)}")

if blocked:
    print(f"\nSTOPPED: AWS blocked access ({blocked}...). Answers so far are cached - re-run later to resume.")

if results:
    decided = [(t, p) for _, t, p in results if p != "unclear"]
    n = len(decided)
    agree = sum(t == p for t, p in decided)
    ct, cp = Counter(t for t, _ in decided), Counter(p for _, p in decided)
    po = agree / n if n else 0
    pe = sum(ct[c] * cp[c] for c in set(ct) | set(cp)) / (n * n) if n else 0
    kap = (po - pe) / (1 - pe) if pe < 1 else 1.0
    truths = Counter(t for _, t, _ in results)
    maj, n_maj = truths.most_common(1)[0]
    print(f"\n{'PARTIAL ' if blocked else ''}SUMMARY  model={MODEL_ID}  locations done {len(results)}/{len(truth)}")
    print(f"Human (consistent labels): yes={truths['yes']}, no={truths['no']}")
    print(f"VLM decided {n}/{len(results)}; agreement {agree}/{n} = {100 * po:.0f}%   kappa {kap:.2f}")
    print(f"Baseline 'always {maj}': {n_maj}/{len(results)} = {100 * n_maj / len(results):.0f}%")
    for t in ("yes", "no"):
        sub = [(tt, p) for tt, p in decided if tt == t]
        if sub:
            print(f"  human {t}: VLM agreed {sum(tt == p for tt, p in sub)}/{len(sub)}")
