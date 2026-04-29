# SetWise MM-Fit Rep Counting Source of Truth

This README is the working reference for SetWise model development on the
MM-Fit dataset. It is grounded in the local prepared dataset, the original
MM-Fit code, and `UbiComp2020.pdf`.

Bottom line: use MM-Fit for exercise classification/segmentation and for
paper-style rep-count evaluation. Do not treat MM-Fit as a standalone neural
rep-regression training set. The rep labels are intentionally centered on 10,
so a naive constant-10 counter is extremely strong on aggregate metrics.

## Source of Truth

Local artifacts:

| Artifact | Path | Purpose |
| --- | --- | --- |
| Prepared watch dataset | `prepared/mmfit_setwise_left_watch_50hz.npz` | Main SetWise tensor export |
| Prepared metadata | `prepared/mmfit_setwise_left_watch_50hz_metadata.json` | Dataset contract and counts |
| Dataset builder | `analysis/prepare_setwise_watch_dataset.py` | Recreates the prepared export |
| Paper | `UbiComp2020.pdf` | Methodology and benchmark reference |
| Raw MM-Fit data | `mm-fit-dataset/w00` through `mm-fit-dataset/w20` | Original per-workout sensor and label files |

Prepared dataset contract:

| Field | Value |
| --- | --- |
| `X` | `(616, 1838, 6)` float32 tensor |
| Sample unit | One manually labeled exercise set |
| Sample rate | `50 Hz`, so each timestep is `0.02s` |
| Channels | `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z` |
| Device | Left smartwatch only |
| Heart rate | Excluded |
| `lengths` | Real sequence length for each set, shape `(616,)` |
| `padding_mask` | Valid timestep mask, shape `(616, 1838)` |
| `y` | Exercise class ID, shape `(616,)`, IDs `0..9` |
| `reps` | Manual set-level repetition count, shape `(616,)` |

Padding matters. The prepared tensor has `621,931` valid timesteps and
`510,277` padded timesteps, so padding is about `45.1%` of the tensor. Sequence
models must use `lengths` or `padding_mask`. Flattened models that standardize
the full padded tensor can learn padding artifacts.

Exercise classes:

| ID | Label | Sets |
| ---: | --- | ---: |
| 0 | `squats` | 64 |
| 1 | `lunges` | 62 |
| 2 | `bicep_curls` | 59 |
| 3 | `situps` | 65 |
| 4 | `pushups` | 65 |
| 5 | `tricep_extensions` | 64 |
| 6 | `dumbbell_rows` | 64 |
| 7 | `jumping_jacks` | 57 |
| 8 | `dumbbell_shoulder_press` | 60 |
| 9 | `lateral_shoulder_raises` | 56 |

Use `split_repo_unseen` as the default split for model development:

| Split | Workouts | Sets |
| --- | --- | ---: |
| Train | `w01, w02, w03, w04, w06, w07, w08, w16, w17, w18` | 301 |
| Validation | `w14, w15, w19` | 86 |
| Test | `w00, w05, w12, w13, w20` | 139 |
| Unused in this split | `w09, w10, w11` | 90 |

The alternative `split_repo_seen` uses `w09, w10, w11` as test workouts and
treats `w00, w05, w12, w13, w20` as unused. Do not mix the two split policies in
one experiment.

## What the Paper Actually Did

MM-Fit contains `616` exercise sets and `6160` repetitions. Participants were
instructed to do `10` repetitions per set; only a small number of sets differ
because of execution or counting variation.

The paper's exercise recognition setup is window-based:

- The recognition task combines activity segmentation and exercise recognition
  by adding a `non_activity` class.
- Sensor windows are `5s` long, or `250` samples at `50 Hz`.
- Evaluation uses overlapping sequential windows with a `0.2s` stride.
- A window is labeled by the majority class in that window.
- The published split assigns whole workouts to train, validation, seen-subject
  test, and unseen-subject test sets.

The paper's repetition counter is not a neural regression head. It assumes the
exercise set has already been segmented and classified, then counts repetitions
with signal processing:

1. Select the segmented set signal for a known or predicted exercise.
2. Standardize the multidimensional signal.
3. Smooth each dimension with a third-degree Savitzky-Golay filter.
4. Project the smoothed multidimensional signal onto the first principal
   component to obtain one 1D signal.
5. Detect local maxima as candidate repetition peaks.
6. Keep high peaks separated by the exercise-specific minimum repetition
   duration.
7. Use autocorrelation around each peak to estimate the local period between
   the exercise-specific minimum and maximum repetition durations.
8. Suppress lower peaks within `0.75 * period`.
9. Remove small peaks below half of the 40th percentile amplitude among
   remaining peaks.
10. Count the remaining peaks.

Paper benchmarks most relevant to SetWise:

| Task | Modality | Reported result |
| --- | --- | --- |
| Exercise recognition, unseen subjects | Left smartwatch acc+gyr | `91.74%` accuracy with pretraining |
| Rep counting | Left smartwatch gyroscope | `0.34` MAE per set |
| Rep counting | Left smartwatch gyroscope | `73.38%` exact, `96.27%` within 1, `98.21%` within 2 |
| Rep counting | Left smartwatch accelerometer | `0.41` MAE per set |

For rep counting, the paper found gyroscope data slightly better than
accelerometer data. Start with left-watch gyroscope, then compare accelerometer
and acc+gyr fusion.

## Rep Label Bias

The rep-count label distribution is heavily biased:

| Reps | Sets | Share |
| ---: | ---: | ---: |
| 1 | 1 | 0.16% |
| 2 | 1 | 0.16% |
| 6 | 2 | 0.32% |
| 7 | 2 | 0.32% |
| 9 | 12 | 1.95% |
| 10 | 566 | 91.88% |
| 11 | 25 | 4.06% |
| 12 | 5 | 0.81% |
| 14 | 2 | 0.32% |

Constant-10 baseline:

| Subset | Sets | MAE | Exact | Within 1 | Within 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All sets | 616 | 0.1396 | 91.88% | 97.89% | 98.70% |
| `split_repo_unseen` train | 301 | 0.1761 | 92.36% | 96.68% | 98.01% |
| `split_repo_unseen` validation | 86 | 0.0930 | 93.02% | 98.84% | 98.84% |
| `split_repo_unseen` test | 139 | 0.1511 | 85.61% | 99.28% | 100.00% |

On the 50 non-10 sets only, constant-10 MAE is `1.72`. The unseen test split has
only 20 non-10 sets, mostly `9` and `11` reps, where constant 10 already gets
`95%` within 1. This means aggregate MAE and off-by-one metrics can make a
trivial model look excellent.

Required rep-count metrics for any learned model:

- Overall MAE, exact, within-1, and within-2.
- The same metrics on non-10 sets only.
- Per-exercise MAE.
- Prediction distribution, including mean and standard deviation.
- Constant-10 baseline on the same split.

If a learned rep model does not beat constant 10 on the same split and does not
improve non-10 behavior, it is not learning useful rep-count structure from
MM-Fit.

## Recommended Modeling Direction

Use a paper-aligned hybrid pipeline:

1. Train or reuse a classifier/segmenter for exercise recognition.
2. Segment the set or infer stable exercise intervals from window predictions.
3. Choose rep-counter thresholds from the predicted exercise class.
4. Count repetitions from the segmented left-watch gyroscope signal.
5. Compare gyroscope, accelerometer, and acc+gyr variants.
6. Report paper-style rep metrics against the manual set labels.

For classification, prefer a windowed model over a full-set flattened MLP if the
goal is to match the paper and support real-time behavior. The paper uses `5s`
windows because they capture repeated movement while keeping recognition delay
reasonable. The prepared set-level `.npz` is still useful, but a windowed
training/evaluation view should be generated from the raw workout streams or
from valid unpadded set ranges.

For rep counting, do not train a rep-regression head only on MM-Fit. The labels
are too concentrated around 10, and the model can minimize loss by predicting
near 10 for everything.

## Initial Rep-Duration Thresholds

These threshold windows are derived from local label durations as
`duration_s / reps`, using the 5th to 95th percentile range padded by 10%. They
are starting values for the paper-style peak/autocorrelation counter, not final
tuned hyperparameters.

| Exercise | Min seconds/rep | Max seconds/rep |
| --- | ---: | ---: |
| `squats` | 1.45 | 2.89 |
| `lunges` | 2.08 | 3.60 |
| `bicep_curls` | 1.31 | 2.79 |
| `situps` | 1.67 | 3.61 |
| `pushups` | 1.11 | 2.63 |
| `tricep_extensions` | 1.11 | 2.69 |
| `dumbbell_rows` | 1.07 | 2.35 |
| `jumping_jacks` | 0.78 | 1.37 |
| `dumbbell_shoulder_press` | 1.43 | 3.15 |
| `lateral_shoulder_raises` | 1.27 | 3.52 |

The paper notes that this style of counter is sensitive to these thresholds and
is not speed invariant. Tune thresholds on the training/validation workouts
only, then report once on the unseen test workouts.

## Notebook and Code Caveats

`SetWise_Kinetic_Profiling_v1.ipynb` is a useful first baseline, but it should
not be used as the model-development target:

- It flattens each full set from `(1838, 6)` to `11028` features.
- It standardizes the flattened padded tensor, including padded regions.
- It trains a multi-task MLP with classification and learned rep regression.
- Its recorded test rep MAE is `2.2669`, far worse than constant 10 at `0.1511`
  on the same unseen test split.
- Its `75.54%` test classification accuracy is not comparable to the paper's
  windowed classification setup.

`SetWise_Kinetic_Profiling_v3.ipynb` contains a better high-level direction
around multi-dataset wrist modeling, but its local MM-Fit loader needs care:

- The recorded run did not load local MM-Fit because it looked for a different
  dataset root layout.
- If loading raw MM-Fit `.npy` streams, slice by frame masks, not by array row
  indices. MM-Fit sensor arrays are `[frame, timestamp_ms, x, y, z]`, and there
  are multiple sensor rows per video frame.
- Mark MM-Fit as classification/segmentation data or rep-counter evaluation
  data, not as primary rep-regression supervision.

The existing `analysis/prepare_setwise_watch_dataset.py` export is correct for
set-level left-watch experiments because it masks by frame range, resamples each
labeled set to `50 Hz`, pads shorter sets, and preserves the real lengths.

## Practical Next Steps

1. Implement the paper-style rep counter as a small, testable module.
2. Evaluate it first with oracle set boundaries and true exercise labels.
3. Evaluate gyroscope-only, accelerometer-only, and acc+gyr variants.
4. Tune exercise thresholds on train/validation workouts only.
5. Add a windowed classifier/segmenter and measure the effect of predicted
   boundaries/classes on rep-count error.
6. Keep the constant-10 baseline in every report.

## Original MM-Fit Setup

This repository contains starter code for the MM-Fit dataset: inertial sensor
data from smartphones, smartwatches, and earbuds, plus time-synchronized RGB-D
video and 2D/3D pose estimates. To download the MM-Fit dataset, visit the
[MM-Fit website](https://mmfit.github.io/).

Conda environment setup:

```bash
conda env create -f environment.yml
conda activate mm-fit
```

Run notebooks:

```bash
jupyter lab
```

Train the original multimodal activity-recognition demo:

```bash
python train_multimodal_ar.py --data "mm-fit/" --num_classes 11 --lr 0.001 --epochs 25 --batch_size 128 --ae_layers 3 --ae_hidden_units 1000 --embedding_units 1000 --ae_dropout 0.0 --window_length 5 --window_stride 0.2 --layers 3 --hidden_units 100 --dropout 0.0
```

## Citation

To cite MM-Fit, use:

David Stromback, Sangxia Huang, and Valentin Radu. 2020. MM-Fit: Multimodal
Deep Learning for Automatic Exercise Logging across Sensing Devices. Proc. ACM
Interact. Mob. Wearable Ubiquitous Technol. 4, 4, Article 168 (December 2020),
22 pages. DOI: https://doi.org/10.1145/3432701

```bibtex
@article{mmfit_2020,
author = {Str\"{o}mb\"{a}ck, David and Huang, Sangxia and Radu, Valentin},
title = {MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices},
year = {2020},
issue_date = {December 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {4},
number = {4},
url = {https://doi.org/10.1145/3432701},
doi = {10.1145/3432701},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = dec,
articleno = {168},
numpages = {22}
}
```
