# Failure analysis: v2 on DEV set (2 Oct 2026)

Reviewed all v2 errors side by side (original vs detector view).

## Root causes
1. White/light vehicles in ROI (bus, SUVs, hatchbacks, vans) - ~5 of 7 dangerous misses
2. Non-road scenes (foliage, grass, dirt, car park) - 4 of 6 "none called lanes"
3. Bonnet/dashboard reflections at bottom of dashcam frames
4. Sky/horizon glare leaking into top of ROI
5. Kerbs/concrete edges fitted as lane lines

## Insight
Light concrete surfaces give low paint contrast even when paint is intact.
This is plausibly a real ADAS risk (camera lane detection also degrades),
not only a detector error. Discuss in report.

## Label ambiguity
Bike path image (pacific_hwy_212441054024464) labelled "ready". Rule from now on:
bike paths, footpaths, laneways = "none". Labels NOT changed after seeing
predictions, to avoid biasing evaluation.

## Plan (ablation)
- v3a: road-presence gate, bonnet cut, sky/foliage masking, kerb-line rejection
- v3b: OpenCV 5 DNN vehicle detection, mask vehicles before paint detection
