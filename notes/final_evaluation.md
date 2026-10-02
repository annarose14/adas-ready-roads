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
