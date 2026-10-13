"""Ablation: does the OpenCV 5 gate matter?
Compares, on the same locations, the same photos and the same blind labels:
  GATED   (v5 as run): only photos OpenCV 5 accepted as road views go to the VLM, best first
  NO GATE: every photo goes to the VLM in capture order, with no OpenCV filtering or ranking
Both arms use the same adaptive rule (ask 2, stop if they agree, else up to 6), the same
OpenStreetMap context and the same decision rules. The NO GATE arm reuses the photo pool the
gated agent built (including photos it fetched because OpenCV found too few road views), which
slightly favours NO GATE. VLM answers are cached by image id, so no photo is paid for twice.
Usage: python ablation_gate.py   (needs AWS credentials for Bedrock)
Output: printed tables and eval_ablation.txt"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from vlm_tool import MODEL_ID, VLMBlocked, ask_lines_visible

SETS = [("Clean test: Penrith, Sutherland, Liverpool, Ryde", "final2_", "labels_final2.csv"),
        ("Final set: 5 suburbs (reused, disclosed)", "finalv5_", "finalv5_labels.csv")]
VLM_MIN, VLM_MAX, VLM_AGREE = 2, 6, 0.67
ROAD = {"painted", "inferred", "no_lanes"}
SYS = {"adas_readable": "yes", "at_risk": "yes", "not_readable": "no",
       "unmarked_by_design": "no", "human_review": "review"}
ROOTS = [p for base in Path(".").glob("data*") if base.is_dir() and base.name != "data_agent"
         for p in base.iterdir() if p.is_dir()]

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def find_image(image_id):
    for d in ROOTS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    p = Path("data_agent") / f"{image_id}.jpg"
    return p if p.exists() else None


def verdict_from(answers):
    yes, no = answers.count("YES"), answers.count("NO")
    decided = yes + no
    if decided == 0:
        return "unclear"
    if yes / decided >= VLM_AGREE:
        return "yes"
    if no / decided >= VLM_AGREE:
        return "no"
    return "split"


def decide(vlm, expected, tag_no):
    """Same rules as agent.py v5, mapped to yes / no / review."""
    if vlm == "yes":
        return "review" if tag_no else "yes"
    if vlm == "no":
        return "no"
    return "review"


def no_gate(frames):
    answers, junk = [], 0
    for f in frames:
        if len(answers) >= VLM_MAX:
            break
        p = find_image(f["image_id"])
        if p is None:
            continue
        a = ask_lines_visible(p, f["image_id"])
        answers.append(a)
        if f["status"] not in ROAD:
            junk += 1
        yes, no = answers.count("YES"), answers.count("NO")
        if len(answers) >= VLM_MIN and ((yes >= VLM_MIN and no == 0) or (no >= VLM_MIN and yes == 0)):
            break
    return verdict_from(answers), len(answers), junk, answers.count("UNCLEAR")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def kappa(ps):
    n = len(ps)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in ps) / n
    ca, cb = Counter(a for a, _ in ps), Counter(b for _, b in ps)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stats(pairs):
    """pairs: (human yes/no/cant_tell, system yes/no/review)"""
    judge = [(h, s) for h, s in pairs if h in ("yes", "no")]
    dec = [(h, s) for h, s in judge if s in ("yes", "no")]
    agree = sum(h == s for h, s in dec)
    rec = {}
    for c in ("yes", "no"):
        sub = [(h, s) for h, s in dec if h == c]
        rec[c] = (sum(h == s for h, s in sub), len(sub))
    parts = [k / n for k, n in rec.values() if n]
    lo, hi = wilson(agree, len(dec))
    return {"n": len(pairs), "decided": sum(s != "review" for _, s in pairs), "agree": agree,
            "n_dec": len(dec), "lo": lo, "hi": hi, "kappa": kappa(dec),
            "balanced": 100 * sum(parts) / len(parts) if parts else float("nan"), "rec": rec}


def fmt(st):
    pct = 100 * st["agree"] / st["n_dec"] if st["n_dec"] else 0
    k = "n/a" if math.isnan(st["kappa"]) else f"{st['kappa']:.2f}"
    return (f"agreement {st['agree']}/{st['n_dec']} = {pct:.0f}% [CI {st['lo']:.0f} to {st['hi']:.0f}%]  "
            f"kappa {k}  balanced {st['balanced']:.0f}%  "
            f"lines found {st['rec']['yes'][0]}/{st['rec']['yes'][1]}  "
            f"no lines found {st['rec']['no'][0]}/{st['rec']['no'][1]}  "
            f"decided {st['decided']}/{st['n']}")


def main():
    out(f"OpenCV gate ablation  |  VLM {MODEL_ID}  |  same photos, labels, map data and rules")
    for title, prefix, labels_file in SETS:
        human = {r["location_id"]: r["label"] for r in csv.DictReader(open(labels_file))}
        rows = {r["segment_id"]: r for r in csv.DictReader(open(f"{prefix}segments_agent.csv"))}
        frames = defaultdict(list)
        for r in csv.DictReader(open(f"{prefix}agent_frames.csv")):
            frames[r["segment_id"]].append(r)
        ctx = {}
        for line in open(f"{prefix}agent_trace.jsonl"):
            t = json.loads(line)
            c = next((s for s in t["trace"] if s["step"] == "road_context"), {})
            ctx[t["segment_id"]] = (c.get("expected_marked"), c.get("lane_markings", "") == "no")
        keys = [k for k in sorted(human) if k in rows]

        gated, ungated = [], []
        g_calls = u_calls = u_junk = u_unclear = pool = pool_junk = 0
        for i, k in enumerate(keys, 1):
            g = SYS[rows[k]["final_verdict"]]
            gated.append((human[k], g))
            g_calls += int(float(rows[k].get("vlm_asked") or 0))
            fr = frames.get(k, [])
            pool += len(fr)
            pool_junk += sum(f["status"] not in ROAD for f in fr)
            v, calls, junk, unclear = no_gate(fr)
            expected, tag_no = ctx.get(k, (None, False))
            ungated.append((human[k], decide(v, expected, tag_no)))
            u_calls += calls
            u_junk += junk
            u_unclear += unclear
            if i % 10 == 0:
                print(f"  ...{i}/{len(keys)} locations", file=sys.stderr)

        out()
        out(f"== {title}  ({len(keys)} locations, {pool} photos in the pool, "
            f"{pool_junk} of them rejected by OpenCV 5 = {100 * pool_junk / max(1, pool):.0f}%) ==")
        out(f"GATED   (v5):  {fmt(stats(gated))}")
        out(f"         VLM calls {g_calls} ({g_calls / len(keys):.1f} per location), "
            f"calls on photos OpenCV rejected: 0")
        out(f"NO GATE:       {fmt(stats(ungated))}")
        out(f"         VLM calls {u_calls} ({u_calls / len(keys):.1f} per location), "
            f"calls on photos OpenCV rejected: {u_junk} ({100 * u_junk / max(1, u_calls):.0f}%), "
            f"UNCLEAR answers: {u_unclear}")
        out(f"Asking about every photo instead would cost {pool} calls ({pool / len(keys):.1f} per location).")

    Path("eval_ablation.txt").write_text("\n".join(lines))
    print("\nSaved eval_ablation.txt")


if __name__ == "__main__":
    try:
        main()
    except VLMBlocked as e:
        sys.exit(f"STOPPED: AWS blocked Bedrock ({e}). Answers so far are cached; run again later.")
