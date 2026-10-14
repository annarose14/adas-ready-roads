# Final evaluation - frozen pipeline v3c-frozen-2026-10-02

## Protocol
- 120 hand-labelled Mapillary images (Sydney: Kensington, Parramatta Rd, Pacific Hwy, Princes Hwy)
- Split: 60 DEV (all tuning and failure analysis), 60 TEST (locked; evaluated ONCE)
- Labels made blind (original images, not detector output)
- Thresholds chosen on DEV from trade-off curve (tune.py): READY_T=75, FAIL_T=70,
  prioritising catching degraded roads over false alarms (safety-first)
- Known label ambiguities documented in failure_analysis_v2.md (bike path,
  side-facing view, traffic occlusion); labels NOT changed after seeing predictions

## DEV results (frozen pipeline)
- Bad roads wrongly called READY: 2/18
- Among detector-graded: bad caught 10/12, false alarms 6/20, agreement 22/32 (69%)
- Sent to review: 7/60
- Good roads skipped (no_lanes/not_road): 7/28

## TEST results
(see eval_test_labels_output_final.txt - run once, not tuned on)

## TEST results (run once, 2 Oct)
- Bad roads wrongly called READY: 7/17 (dev: 2/18)
- Among detector-graded: bad caught 2/9 (dev 10/12), false alarms 7/21 (dev 6/20),
  agreement 15/30 = 50% (dev 69%)
- "fail" roads: 5/5 went to no_lanes (2) or review (3) - none called ready
- Sent to review: 6/60; good roads skipped: 8/29; none-scenes correctly skipped 6/13

## Interpretation
Single-image ready/degraded thresholds OVERFIT the 60-image dev set
(~8 tuning iterations, only 12 bad dev examples). Segmentation, ego-vehicle cut,
road gate and review routing generalised; per-image contrast grading did not.
This test set is now USED: no further changes judged against it.

## Response (design change, not threshold tweaking)
- Move from per-image to per-road-segment verdicts (consensus over neighbouring
  frames in the same Mapillary sequence) - the agent loop.
- Treat persistent no_lanes on a road expected to be marked as a fail signal.
- Final evaluation on a FRESH test set from new, unseen suburbs.

## DNN engine change for deployment (3 Oct)
OpenCV 5 new DNN engine segfaults on this model on Linux/ARM64 (AWS Lambda), reproducibly
at net.forward(); classic engine stable (~300 ms/frame, ARM64). Deployed with ENGINE_CLASSIC.
compare_engines.py: classic vs new engine verdicts agree on 318/320 usable images (99.4%);
2 differences are threshold edge cases. Reported evaluation numbers were made with the new engine.

## AWS deployment (3 Oct)
- Lambda (container image, ARM64/Graviton, 2048 MB, Sydney ap-southeast-2), ECR, Function URL (public, CORS)
- IAM: function role = logs only (AWSLambdaBasicExecutionRole); URL invoke restricted via InvokedViaFunctionUrl
- Mapillary token stored as function environment setting (not in code/git)
- Verified: upload + mapillary_id both return identical verdict to local pipeline (ready, contrast 86.3)
- Latency: cold start ~14.7 s round trip; warm ~2.7 s processing/photo at 2048 MB (Mac: ~0.5 s)

## HIGH-priority validation by photo review (3 Oct, view_segments.py)
6 HIGH segments = only 2 locations (overlapping segments from multiple sequences).
- Anzac Pde S0006: wide junction + roadworks; dashed lines clearly visible -> detector limitation
- Anzac Pde S0007: lines visible; tram tracks create false paint; vans occlude -> limitation
- Anzac Pde S0014: large bonnet, tiny ROI; lines visible -> limitation
- Anzac Pde S0038: 3 of original frames are PEDESTRIAN shop-window/footpath photos counted as
  "no lanes" -> BUG; fetched frames dominated by light-rail tram
- Railway Rd S0070: pale concrete junction, faint markings -> plausible GENUINE finding
  (low paint/surface contrast, consistent with concrete insight)
- Railway Rd S0071: lines visible; segmentation labelled sunlit road as vehicle -> limitation
Conclusion: HIGH flags mostly false alarms from complex urban scenes. Fixes planned:
(1) vehicle-speed filter on sequences, (2) OSM junction detection -> review,
(3) spatial merging of duplicate segments. Remaining: tram tracks, roadworks, glare = limitations.

## Location-level TEST (fresh suburbs Burwood + Bondi Junction, 4 Oct, blind labels)
36 locations. System decided 24 (67%), 12 to review (human could judge all 12).
Agreement on decided: 10/24 = 42% (baseline always-readable 50%).
Problem roads decided: 6 -> caught 2, DANGEROUS (called readable) 4.
Readable roads: 12 -> false alarms 7. Unmarked: 11 -> system agreed 3.
Interpretation: safeguards (junction review, merging, vehicle filter) generalise; the
hand-built frame-level lane finder does NOT generalise to unseen suburbs.
Caveats: small n; labels made late at night (labels NOT changed after seeing results).
This TEST set is now USED.

## v4 hybrid (UFLD lane network + paint verification), dev2 = former test suburbs (4 Oct)
Location IDs drifted between runs (OSM merges) -> labels re-mapped by shared photos (remap_labels.py).
v4 on dev2: exact 38%, marked-vs-unmarked 76%. v3c on same: exact 42%, marked-vs-unmarked ~88%.

## Label reliability (intra-rater, same 36 locations, two blind passes, shuffled)
All labels: 53% (kappa 0.24). Marked vs unmarked: 78% (kappa 0.38).
Readable vs at-risk (marked only): 62% (kappa 0.14 = poor).
Conclusion: paint-wear grading from crowd photos is NOT reliably judgeable even by a human;
it cannot be used as an evaluation target. Final evaluation uses the reliable question:
"are painted lane lines visible?" (yes / no / can't tell). v4 FROZEN for final test.

## FINAL TEST (v4 frozen; 5 fresh suburbs: Chatswood, Hurstville, Hornsby, Parramatta CBD, Bankstown)
48 locations, blind binary labels ("painted lane lines visible?"), sheets generated from the evaluated run.
Human: yes=39, no=9. System decided 39/48.
System agreement 17/39 = 44% [95% CI 29-59%], kappa -0.01 (chance). Baseline always-yes 81%; OSM-only 53%.
Diagnosis (post-hoc, read-only, no re-scoring):
- 13/19 human-yes/system-no misses: OSM minor-road rule + no evidence fetching on minor roads.
- Fundamental: painted-lane rate overlaps between human yes/no (11/32 'yes' locations at 0.00 incl.
  Marsden St 69 frames; 3/7 'no' at 0.37-0.64). No threshold beats the majority baseline (~64% best).
Conclusion: detector work STOPPED. Lane-marking visibility from crowd-sourced imagery does not
generalise with UFLD+paint verification or the hand-built v3c detector. Submission reports this as a
rigorous negative result alongside the agent, OpenCV 5 and AWS engineering.

## v5 (OpenCV-gated VLM tool, adaptive asking) on DEV (former test suburbs, 4 Oct)
29 locations, labels = first-pass location labels mapped to yes/no (incl. inconsistent ones).
Agreement 22/27 = 81% [63-92], kappa 0.54. Human yes found 17/19; human no 5/8.
Baselines: always-yes 69%, OSM-only 76% (kappa 0.38). 2.3 VLM questions/location.
v5 FROZEN for evaluation.

## v5 on the 48-location final set (RE-USED; v5 design partly informed by its v4 diagnosis - disclosed)
Agreement 35/45 = 78% [64-87], kappa 0.31; yes found 31/38, no found 4/7; 3 sent to review.
Baselines: always-yes 81% (balanced 50%); OSM-only 54% (kappa -0.05). v4 on same set: 44%, kappa -0.01.
Balanced accuracy added to the metric set BEFORE the clean Penrith/Sutherland test.

## Clean test B setup (Penrith, Sutherland, Liverpool, Ryde - never used before)
Frozen v5. 7 of 56 OSM lookups failed twice (Overpass overload). In v5 OSM does not decide
the lines-visible verdict (VLM evidence does); it affects priority, unmarked-vs-not-readable
wording, map-conflict check and location grouping only. Disclosed; not re-run further.
Consensus subset (54 locations both passes = yes): confirmed 42, review 6, wrongly 'no' 6
-> 88% of decided; 11% routed to humans; 11% false 'no lines' (false alarms, not dangerous misses).

## Ablation: OpenCV gate vs no gate (13 Oct, ablation_gate.py, eval_ablation.txt)
Same locations, photo pools, labels, OSM context and rules; cached VLM answers.
OpenCV rejected 7% (clean test) and 9% (final set) of pooled photos.
Clean test: gated 42/49 = 86%, no gate 41/50 = 82%. Final set: gated 35/45 = 78% (balanced 69%),
no gate 33/44 = 75% (balanced 72%). Without the gate 5% and 12% of calls went to rejected photos.
Conclusion: the gate saves wasted calls but its accuracy effect is small and within noise; the
vehicle speed filter already removes most unusable photos. Reported as found.

## Road focus probe: does an OpenCV 5 road crop help the VLM? (14 Oct, crop_probe.py, development data only)
Same photos and prompt; modes full / crop (road region from segmentation) / mask (non-road painted grey).
Pooled development (72 locations): full balanced 78% (lines 51/61, no lines 8/11); crop 77% (51/62, 5/7);
mask 74% (35/57, 6/7, 12 unclear). Pre-agreed rule (>= +5 balanced) not met -> not adopted, no fresh test.
Interpretation: the VLM uses scene context (kerbs, road width, traffic) to judge whether lines belong there.

## Missing-markings test: plan written BEFORE running (14 Oct)
Purpose: measure whether frozen v5 finds roads WITHOUT visible lane lines (unproven so far: 9 of 16).
Areas never used before, chosen because unmarked or faded roads are likely (sample enriched, disclosed):
Paddington, Oran Park, Marsden Park, Smithfield industrial. System: v5 exactly as deployed, no changes.
Blind binary labels on sheets generated from this exact run. Evaluated once.
Main numbers: no lines found (recall on human 'no'), lines found (recall on human 'yes'),
balanced accuracy, and precision of the system's 'no' answers. Reported whatever they show.

## Step 6 run log (14 Oct 2026, before labelling)
- Fetched paddington, oran_park, marsden_park, smithfield into data_final3: 583 images.
- Pipeline v4 (unchanged): painted 215, inferred 132, no_lanes 128, not_road 24, review 11, unusable 73.
- Agent v5 (unchanged logic): kept 496/510 frames, 86 segments merged into 60 locations.
- Public Overpass servers stopped responding partway through the run (a request hung on a socket read).
  agent_tools.py now rotates across Overpass servers with a 60 s limit per location.
  The final run used run_agent_cached_osm.py: cached OSM answers only, the rest recorded as osm_error.
  osm_error rows: 32. OSM is context only in v5 and does not change the verdict.
- 60 blind contact sheets written to blind_final3/. Verdicts not viewed before labelling.

## Step 7 plan (14 Oct 2026, written before any analysis)
Goal: fewer false "lines visible" on roads without lines.
- 7a Error analysis on final3 (already evaluated once; its result stays as reported). Diagnosis only.
- 7b Candidate fixes tried on development + final2 + final3 data only. Adopt one only if balanced
  accuracy rises by at least 5 points AND lines confirmed stays at 85% or more. At most 3 candidates.
- 7c The adopted version is frozen, then tested once on 4 new suburbs, labelled blind.

## Step 7a finding and second labelling pass (14 Oct 2026)
- Looking at the 8 "no lines" locations the agent called "lines visible", most sheets show painted
  lines (double centre lines, dashed lanes). The first pass was done at 2 am in a small window.
- Correcting only those 8 would bias the result, so all 60 locations are relabelled in a second,
  shuffled, full screen pass (labels_final3_pass2.csv). Both passes and their agreement are reported.
- Disclosed: the labeller had seen those 8 sheets during error analysis before the second pass.
- Pass 2 result: pass 1 vs pass 2 agree 52/60 (kappa 0.69); all 8 changes were no -> yes, 7 of them
  sheets seen in error analysis. Against pass 2: 55/56 decided correct (98%, kappa 0.95), lines 44/44,
  no lines 11/12. Only remaining error: L0043. Both passes reported; pass 1 stays the blind result.
- Decision: no agent change (one real error does not justify one). v5 stays frozen.
- Next: a new clean test on 4 suburbs, labelled carefully full screen, before results are viewed.
