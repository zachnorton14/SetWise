from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ACTIONS = [
    "squats",
    "lunges",
    "bicep_curls",
    "situps",
    "pushups",
    "tricep_extensions",
    "dumbbell_rows",
    "jumping_jacks",
    "dumbbell_shoulder_press",
    "lateral_shoulder_raises",
]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
FEATURE_NAMES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
]

# Repo split used in train_multimodal_ar.py
TRAIN_WORKOUTS = {"w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"}
VAL_WORKOUTS = {"w14", "w15", "w19"}
SEEN_TEST_WORKOUTS = {"w09", "w10", "w11"}
UNSEEN_TEST_WORKOUTS = {"w00", "w05", "w12", "w13", "w20"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined set-level smartwatch dataset from MM-Fit."
    )
    parser.add_argument(
        "--dataset-dir",
        default="mm-fit-dataset",
        help="Path to the extracted MM-Fit dataset directory.",
    )
    parser.add_argument(
        "--watch-side",
        choices=("left", "right"),
        default="left",
        help="Which smartwatch side to export.",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=50.0,
        help="Target resampling rate in Hz for each labeled set.",
    )
    parser.add_argument(
        "--output-dir",
        default="prepared",
        help="Directory where the combined dataset files will be written.",
    )
    return parser.parse_args()


def workout_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.iterdir() if path.is_dir())


def load_labels(label_path: Path) -> list[tuple[int, int, int, str]]:
    labels = []
    with label_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            labels.append((int(row[0]), int(row[1]), int(row[2]), row[3]))
    return labels


def split_name(workout_id: str, seen_test: bool) -> str:
    if workout_id in TRAIN_WORKOUTS:
        return "train"
    if workout_id in VAL_WORKOUTS:
        return "val"
    if seen_test:
        return "test" if workout_id in SEEN_TEST_WORKOUTS else "unused"
    return "test" if workout_id in UNSEEN_TEST_WORKOUTS else "unused"


def resample_channels(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.shape[0] == target_length:
        return values.astype(np.float32, copy=False)

    if values.shape[0] == 1:
        return np.repeat(values.astype(np.float32), target_length, axis=0)

    src_positions = np.linspace(0.0, 1.0, num=values.shape[0], endpoint=True)
    dst_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    resampled = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        resampled[:, col] = np.interp(dst_positions, src_positions, values[:, col])
    return resampled


def extract_set_tensor(
    acc_data: np.ndarray,
    gyr_data: np.ndarray,
    start_frame: int,
    end_frame: int,
    target_rate: float,
) -> tuple[np.ndarray, int, float]:
    frame_duration_s = max((end_frame - start_frame) / 30.0, 1.0 / target_rate)
    target_length = max(1, int(round(frame_duration_s * target_rate)))

    acc_mask = (acc_data[:, 0] >= start_frame) & (acc_data[:, 0] <= end_frame)
    gyr_mask = (gyr_data[:, 0] >= start_frame) & (gyr_data[:, 0] <= end_frame)

    acc_values = acc_data[acc_mask, 2:5]
    gyr_values = gyr_data[gyr_mask, 2:5]

    if acc_values.shape[0] == 0 or gyr_values.shape[0] == 0:
        raise ValueError(
            f"No smartwatch samples found for frame range [{start_frame}, {end_frame}]"
        )

    acc_resampled = resample_channels(acc_values, target_length)
    gyr_resampled = resample_channels(gyr_values, target_length)
    combined = np.concatenate([acc_resampled, gyr_resampled], axis=1)
    return combined, target_length, frame_duration_s


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    side_prefix = "sw_l" if args.watch_side == "left" else "sw_r"
    workouts = workout_dirs(dataset_dir)

    sequences = []
    lengths = []
    labels = []
    reps = []
    workout_ids = []
    set_indices = []
    durations_s = []
    frame_ranges = []
    split_seen = []
    split_unseen = []

    for workout in workouts:
        workout_id = workout.name
        acc_path = workout / f"{workout_id}_{side_prefix}_acc.npy"
        gyr_path = workout / f"{workout_id}_{side_prefix}_gyr.npy"
        label_path = workout / f"{workout_id}_labels.csv"

        if not acc_path.exists() or not gyr_path.exists():
            raise FileNotFoundError(
                f"Missing smartwatch files for {workout_id}: {acc_path.name}, {gyr_path.name}"
            )

        acc_data = np.load(acc_path)
        gyr_data = np.load(gyr_path)
        label_rows = load_labels(label_path)

        for set_idx, (start_frame, end_frame, rep_count, action_name) in enumerate(label_rows):
            if action_name not in ACTION_TO_INDEX:
                continue

            sequence, seq_len, duration_s = extract_set_tensor(
                acc_data=acc_data,
                gyr_data=gyr_data,
                start_frame=start_frame,
                end_frame=end_frame,
                target_rate=args.target_rate,
            )
            sequences.append(sequence)
            lengths.append(seq_len)
            labels.append(ACTION_TO_INDEX[action_name])
            reps.append(rep_count)
            workout_ids.append(workout_id)
            set_indices.append(set_idx)
            durations_s.append(duration_s)
            frame_ranges.append((start_frame, end_frame))
            split_seen.append(split_name(workout_id, seen_test=True))
            split_unseen.append(split_name(workout_id, seen_test=False))

    if not sequences:
        raise RuntimeError("No labeled smartwatch sets were exported.")

    max_length = max(lengths)
    num_samples = len(sequences)
    data = np.zeros((num_samples, max_length, len(FEATURE_NAMES)), dtype=np.float32)
    padding_mask = np.zeros((num_samples, max_length), dtype=np.uint8)
    for idx, sequence in enumerate(sequences):
        data[idx, : sequence.shape[0], :] = sequence
        padding_mask[idx, : sequence.shape[0]] = 1

    labels_arr = np.asarray(labels, dtype=np.int64)
    reps_arr = np.asarray(reps, dtype=np.int64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    durations_arr = np.asarray(durations_s, dtype=np.float32)
    frame_ranges_arr = np.asarray(frame_ranges, dtype=np.int64)
    workout_ids_arr = np.asarray(workout_ids)
    set_indices_arr = np.asarray(set_indices, dtype=np.int64)
    split_seen_arr = np.asarray(split_seen)
    split_unseen_arr = np.asarray(split_unseen)
    one_hot = np.eye(len(ACTIONS), dtype=np.uint8)[labels_arr]

    filename_stem = f"mmfit_setwise_{args.watch_side}_watch_{int(args.target_rate)}hz"
    npz_path = output_dir / f"{filename_stem}.npz"
    metadata_path = output_dir / f"{filename_stem}_metadata.json"

    np.savez_compressed(
        npz_path,
        X=data,
        lengths=lengths_arr,
        padding_mask=padding_mask,
        y=labels_arr,
        y_onehot=one_hot,
        reps=reps_arr,
        workout_id=workout_ids_arr,
        set_index=set_indices_arr,
        duration_s=durations_arr,
        frame_range=frame_ranges_arr,
        split_repo_seen=split_seen_arr,
        split_repo_unseen=split_unseen_arr,
        feature_names=np.asarray(FEATURE_NAMES),
        label_names=np.asarray(ACTIONS),
        watch_side=np.asarray(args.watch_side),
        target_rate_hz=np.asarray(args.target_rate, dtype=np.float32),
    )

    label_counts = {action: int(np.sum(labels_arr == idx)) for idx, action in enumerate(ACTIONS)}
    rep_stats = {
        "min": int(reps_arr.min()),
        "max": int(reps_arr.max()),
        "mean": float(reps_arr.mean()),
    }
    duration_stats = {
        "min_seconds": float(durations_arr.min()),
        "max_seconds": float(durations_arr.max()),
        "mean_seconds": float(durations_arr.mean()),
        "max_timesteps": int(max_length),
    }
    split_counts = {
        "repo_seen": {name: int(np.sum(split_seen_arr == name)) for name in np.unique(split_seen_arr)},
        "repo_unseen": {name: int(np.sum(split_unseen_arr == name)) for name in np.unique(split_unseen_arr)},
    }
    metadata = {
        "dataset_path": str(npz_path),
        "watch_side": args.watch_side,
        "target_rate_hz": args.target_rate,
        "num_samples": num_samples,
        "tensor_shape": list(data.shape),
        "feature_names": FEATURE_NAMES,
        "label_names": ACTIONS,
        "label_counts": label_counts,
        "rep_stats": rep_stats,
        "duration_stats": duration_stats,
        "split_counts": split_counts,
        "notes": [
            "Each sample is one labeled exercise set from MM-Fit.",
            "Features are [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] from a single smartwatch.",
            "Sequences are padded to the maximum set length; use lengths or padding_mask during training.",
            "split_repo_seen and split_repo_unseen mirror the workout splits used by the repo.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote dataset: {npz_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Samples: {num_samples}")
    print(f"Tensor shape: {tuple(data.shape)}")
    print(f"Max timesteps: {max_length}")
    print(f"Label counts: {label_counts}")
    print(f"Repo seen split counts: {split_counts['repo_seen']}")
    print(f"Repo unseen split counts: {split_counts['repo_unseen']}")


if __name__ == "__main__":
    main()
