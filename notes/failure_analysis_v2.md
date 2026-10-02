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

## v3a result: colour road gate REJECTED (2 Oct)
tune_road.py on DEV: road_frac of real roads ranges 0.00-0.98, non-roads 0.10-1.00.
Full overlap; no threshold separates them (e.g. 0.15 loses 6/46 roads, keeps 7/9 non-roads).
Causes: camera colour casts, shade, vehicles covering ROI. Two "none" images are
unmarked roads (laneway) - correct verdict is no_lanes, not not_road.
Gate disabled. Plan: OpenCV 5 DNN semantic segmentation (road/vehicle/sky/vegetation).

## Stability note (2 Oct)
lane_check_v3b.py crashed once with "segmentation fault" (inside OpenCV DNN, C++ level),
then ran fine on re-run with identical code/data -> intermittent, likely threading in
OpenCV 5 new DNN engine. TODO before AWS: test ENGINE_CLASSIC or cv.setNumThreads(1).
