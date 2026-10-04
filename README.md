# ADAS-Ready Roads

**Can a camera-based lane-keeping system see the painted lane lines on this road?**

ADAS-Ready Roads is an agent that screens Sydney streets for lane-marking visibility using
crowd-sourced street-level imagery (Mapillary), **OpenCV 5**, OpenStreetMap context and a
vision-language model on **Amazon Bedrock** - and routes uncertain places to people.
It is built for councils that want to know where driver-assist cameras may struggle,
without sending out survey vehicles.

> **Research prototype.** Verdicts are screening hints to prioritise human inspection, not
> safety certifications. The evaluation below states what has and has not been demonstrated.

- **Live dashboard:** https://d1dzzj0gvxxl4o.cloudfront.net
- **Live API (AWS Lambda Function URL):** `POST` JSON `{"mapillary_id": "<id>"}` or `{"image_base64": "<jpeg>"}`
  to https://unft6yl7d6znbr6gvfvqbwzqmq0tidph.lambda-url.ap-southeast-2.on.aws/
- Entry for the OpenCV AI Competition 2026 (powered by AWS).

---

## Results at a glance

Question asked of every location: *are painted lane lines visible?* (blind human labels)

| Evaluation | Locations | Lines visible -> confirmed | No lines -> found |
|---|---|---|---|
| Clean test, 4 never-used suburbs (frozen v5) | 56 | **42/48 decided = 88%** [95% CI 75-94%] | 0/1 (only one such road) |
| Final set, 5 suburbs (re-used; disclosed) | 48 | 31/38 = 82% | 4/7 |
| Development suburbs | 29 | 17/19 = 89% | 5/8 |
| **Pooled** | 133 | **90/105 ~ 86%** | **9/16 ~ 56% (very uncertain)** |

- About 1 in 10 locations is routed to human review instead of being guessed.
- **Detecting *missing* markings is not yet demonstrated**: roads without lines are rare in
  crowd-sourced imagery (16 of 133 labelled locations).
- Earlier OpenCV-only versions (hand-built paint detector; lane network + paint check) scored
  **at chance** on unseen suburbs (e.g. 44%, kappa -0.01 on the 48-location set; v5: 78%, kappa 0.31).
- OpenStreetMap alone is a poor guide (54% on the 48-location set).
- Human self-agreement on **paint wear** was poor (kappa 0.14, two blind passes), so the
  system does **not** grade wear - it answers the measurable question of visibility.

Full numbers, protocol and history: `notes/final_evaluation.md`, `eval_*.txt`.

---

## How it works

~~~
Mapillary photos (CC BY-SA)                         OpenStreetMap (ODbL)
        |                                                   |
        v                                                   |
 OpenCV 5 pipeline (every photo)                            |
   quality gate (dark / blurry / panorama)                  |
   fastseg semantic segmentation (OpenCV DNN):              |
     road gate, own-bonnet removal, vehicles                |
   Ultra-Fast-Lane-Detection (OpenCV DNN) + paint check     |
        |                                                   |
        v                                                   v
 Agent (per location)  PERCEIVE -> CONTEXT -> ACT -> ASK -> DECIDE -> ESCALATE
   - vehicle-speed filter (drops walking / handheld sequences)
   - merges overlapping segments on the same road into one location
   - ACT: fetches more Mapillary photos if < 3 usable road photos
   - ASK: Tool C, Amazon Bedrock (Nova Lite) on the best OpenCV-approved photos,
          adaptively: ask 2, stop if they agree, else up to 6 (avg ~2.4 per location)
   - DECIDE with OSM: lines visible -> readable | no lines + marked road -> NOT readable
                      (HIGH priority on major roads) | no lines + minor road -> unmarked
   - ESCALATE: split evidence, no usable photos, or map conflict -> human review
   - every step logged to *_agent_trace.jsonl; every photo + answer to *_agent_frames.csv
        |
        v
 AWS: Lambda (container, ARM64/Graviton, Sydney) -> live single-photo API
      S3 (private) + CloudFront (OAC) -> public dashboard
      IAM least privilege: logs only + bedrock:InvokeModel on Nova Lite only
~~~

### Why a vision-language model?
Three OpenCV-only approaches were built and honestly tested on unseen suburbs; all were at
chance (see History). The evaluation target - "are painted lines visible?" - is a human visual
judgement, and a VLM matched blind human labels far better (dev probe 96%, kappa 0.84 on
consistently labelled locations). OpenCV 5 remains the first stage: it rejects unusable and
non-road photos and selects the frames the VLM sees, which keeps calls few and cheap.

### OpenCV 5 findings
- **The OpenCV 5 new DNN engine segfaults** on both ONNX models on Linux/ARM64 (AWS Lambda),
  reproducibly at `net.forward()`, and crashed intermittently on macOS. The **classic engine**
  (`cv.dnn.readNetFromONNX(path, engine=cv.dnn.ENGINE_CLASSIC)`) is stable. Classic vs new
  engine verdicts agreed on 318/320 photos (`compare_engines.py`).
- `HoughLinesP` output shape changed from 4.x; handled with `reshape(-1, 4)`.

---

## Repository guide

| File | Purpose |
|---|---|
| `adas_pipeline.py` | OpenCV 5 per-photo pipeline v4 (quality, segmentation, lane network + paint check) |
| `lane_features.py` | Ultra-Fast-Lane-Detection wrapper + paint verification |
| `agent.py` | Location agent v5 (perceive, context, fetch, ask VLM, decide, escalate, traces) |
| `agent_tools.py` | Tool A `fetch_frames` (Mapillary), Tool B `road_context` (OSM Overpass, cached) |
| `vlm_tool.py` | Tool C: Amazon Bedrock Nova Lite, "are painted lane lines visible?" (cached) |
| `segments.py` | Vehicle-speed filter, ~100 m segments, per-segment summary |
| `run_pipeline.py` | Batch-runs the pipeline over `<data>/<area>/*.jpg` |
| `fetch_mapillary.py` | Downloads images + metadata (incl. photographer credit) for a bounding box |
| `handler.py`, `Dockerfile` | AWS Lambda single-photo endpoint (OpenCV 5 + Bedrock) |
| `dashboard/` | Static map dashboard (Leaflet), hosted on S3 + CloudFront |
| `view_segments.py`, `label_binary.py`, `label_locations.py` | Contact sheets and blind labelling tools |
| `evaluate_binary.py`, `evaluate_locations.py`, `label_agreement.py`, `remap_labels.py` | Evaluation, reliability, label re-mapping |
| `export_seg.py`, `export_ufld.py` | One-time ONNX export of the two models |
| `notes/` | Failure analyses and the full evaluation log |

---

## Run it yourself

**1. Setup** (macOS/Linux, Python 3.12+):
~~~bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
~~~

**2. Models** (not in git; see licences below):
~~~bash
pip install -r requirements-export.txt
python export_seg.py           # -> models/fastseg_large_512x1024.onnx
# Ultra-Fast-Lane-Detection: clone https://github.com/cfzd/Ultra-Fast-Lane-Detection to ../ufld,
# download culane_18.pth from that README into models/, then:
python export_ufld.py          # -> models/ufld_culane18_288x800.onnx (loaded with weights_only=True)
~~~

**3. Credentials:** a Mapillary token in `MAPILLARY_TOKEN`; AWS credentials with Bedrock access
(`aws login --region ap-southeast-2`) for the VLM tool.

**4. Fetch, analyse, run the agent:**
~~~bash
python fetch_mapillary.py kensington 151.215 -33.925 151.230 -33.910 data 100
python run_pipeline.py data output_main
python agent.py output_main/results.csv main_
python view_segments.py --prefix main_ all            # annotated contact sheets
~~~
Outputs: `main_segments_agent.csv/.geojson`, `main_review_queue.csv`, `main_agent_trace.jsonl`,
`main_agent_frames.csv`.

**5. Deploy** (summary): build the container (`docker build --platform linux/arm64 --provenance=false
--sbom=false`), push to ECR, create a Lambda (ARM64, 3008 MB, 60 s) from the image with a role that
has `AWSLambdaBasicExecutionRole` + `bedrock:InvokeModel` on Nova Lite only, add a Function URL
(both `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction` with `--invoked-via-function-url`),
and host `dashboard/` on a private S3 bucket behind CloudFront with Origin Access Control.

---

## History (what was tried, and what the tests said)

| Version | Idea | Honest result |
|---|---|---|
| v0-v3c | Hand-built OpenCV paint detector (top-hat, Hough, contrast) + segmentation | Overfit dev; locked test 7/17 dangerous misses; fresh-suburb location test 42% |
| v4 | Ultra-Fast-Lane-Detection (OpenCV DNN) + paint verification | Final test 44%, kappa -0.01 (chance) |
| v5 | OpenCV-gated VLM tool on Bedrock + adaptive evidence + OSM | Clean test: 88% of lined locations confirmed; missing markings unproven |

Protocol: thresholds tuned on development data only; each test set used once (re-use disclosed);
blind labels; contact sheets generated from the exact run being evaluated; label reliability
measured with a second blind pass.

---

## Limitations and responsible use
- Screening only - never a substitute for an on-site inspection or a safety assessment.
- Missing-marking detection is not yet validated (too few unmarked roads in the data).
- Labels come from a single labeller; wear cannot be judged reliably from crowd photos.
- Crowd imagery is uneven in time, camera, weather and coverage; results reflect when photos were taken.
- OpenStreetMap lookups can fail under server load (disclosed per evaluation).
- Photographer credit and licence are kept with every photo; no images are stored in the repo.

## Credits and licences
- Street imagery: Mapillary contributors, CC BY-SA 4.0 (creator recorded per photo).
- Road data and map tiles: (c) OpenStreetMap contributors, ODbL.
- fastseg MobileV3Large (MIT), Cityscapes-trained weights.
- Ultra-Fast-Lane-Detection, Qin et al., ECCV 2020 (MIT), CULane-trained weights.
- Amazon Nova Lite via Amazon Bedrock.
- Built with OpenCV 5, AWS Lambda, ECR, S3, CloudFront, IAM and Bedrock.

Author: Anna Rose Bijoy - Master of IT (AI), UNSW Sydney.
