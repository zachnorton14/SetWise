# SetWise Changelog

## v3 (current)
- INSIGHT-LME standalone notebook (`INSIGHT-LME/INSIGHT_LME_Processing.ipynb`): EDA + save raw 50 Hz segments to `INSIGHT_LME_processed.npz`; `load_insight()` in v3 now loads from .npz instead of re-parsing the CSV (fixes FileNotFoundError, segments per-trial instead of per-participant).
- Single-branch time-normalized architecture (SetWiseV3). Dropped variable-length masked pooling; all inputs resampled to T_NORM=512 via scipy_resample. Rep count encoded as cycle frequency.
- Fixed recofit -1 sentinel bug: was leaking 3 339 non-exercise rows into rep regression as rep_count=-1. Guard is now `reps > 0`.
- INSIGHT-LME promoted to `has_reps=True`: whole-set resampling preserves cycle count regardless of per-rep normalisation.
- recofit unlocked for rep regression (cycle counting transfers across sensor placements): 1 011 annotated sets, range 1–61.

## v2 Architecture & Training
- Switched to dual-branch model: natural-timing branch (variable-length, masked pooling) + time-normalized branch (fixed T_NORM=512, cycle frequency encodes rep count). Rep counting is primary task; classification is auxiliary.
- Loss weights: REP_WEIGHT=1.0 (primary), CLS_WEIGHT=0.1 (auxiliary).
- Two-phase training retained: pretrain on recofit+MyoGym (forearm), fine-tune on wrist data.

## Dataset Decisions
- **INSIGHT-LME** (`has_reps=True`): time-normalized per rep, so cycle count in the T_NORM branch accurately encodes rep count. Contributes to both classification and rep regression.
- **mm-fit** (`has_reps=False`): 92% of sets are exactly 10 reps — too narrow to train rep regression without introducing severe bias. Classification only.
- **Whales1and2** (`has_reps=True`): rep counts range 2–30, wrist sensor, primary real-time rep regression source.
- **recofit** (`has_reps=True` where annotated): forearm, pretrain only.
- **MyoGym** (`has_reps=False`): forearm, no rep annotations, pretrain classification only.

## Sensor / Deployment
- Final product is watch-based wrist IMU. Wrist datasets used for fine-tuning; forearm datasets for pretraining only.
- All signals downsampled to TARGET_HZ=50. Natural branch padded to MAX_LEN=2000; normalized branch resampled to T_NORM=512.
