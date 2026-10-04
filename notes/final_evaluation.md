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
