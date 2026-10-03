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
