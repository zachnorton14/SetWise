#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/SetwiseKineticDatasets")
DEFAULT_MMFIT_DIR = DEFAULT_DRIVE_ROOT / "mm-fit-dataset"
DEFAULT_WHALES_DIR = DEFAULT_DRIVE_ROOT / "Whales1and2_Raw_Labelled"
DEFAULT_OUTPUT_DIR = DEFAULT_DRIVE_ROOT / "prepared"
LOCAL_MMFIT_DIR = Path("mm-fit/mm-fit-dataset")
LOCAL_WHALES_DIR = Path("Whales1and2_Raw_Labelled")
LOCAL_OUTPUT_DIR = Path("prepared")

G_TO_M_PER_S2 = 9.80665
FEATURE_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

MMFIT_ACTIONS = [
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
MMFIT_ACTION_TO_INDEX = {name: idx for idx, name in enumerate(MMFIT_ACTIONS)}
MMFIT_CANONICAL = {
    "dumbbell_rows": "rows",
    "dumbbell_shoulder_press": "shoulder_press",
    "lateral_shoulder_raises": "lateral_raises",
}

WHALES_CANONICAL = {
    "30BP": "bench_press",
    "30DBP": "bench_press",
    "45DBP": "bench_press",
    "AIDBC": "bicep_curls",
    "APULL": "pullups",
    "CGCR": "rows",
    "CGOCTE": "tricep_extensions",
    "CRDP": "rear_delt",
    "DLR": "lateral_raises",
    "DSP": "shoulder_press",
    "DWC": "wrist_curl",
    "FAPU": "face_pull",
    "HT": "hip_thrust",
    "IDBC": "bicep_curls",
    "ILE": "leg_extension",
    "LE": "leg_extension",
    "LHC": "hamstring_curl",
    "MGTBR": "rows",
    "MIBP": "bench_press",
    "MPBC": "bicep_curls",
    "MRF": "rear_delt",
    "MSP": "shoulder_press",
    "MTE": "tricep_extensions",
    "NGCR": "rows",
    "PREC": "bicep_curls",
    "PULL": "pullups",
    "PUSH": "pushups",
    "SAOCTE": "tricep_extensions",
    "SAODTE": "tricep_extensions",
    "SACLR": "lateral_raises",
    "SAP": "pullups",
    "SBCTP": "tricep_extensions",
    "SBLP": "pullups",
    "SECR": "calf_raise",
    "SHC": "hamstring_curl",
    "SHSS": "squats",
    "SMS": "squats",
    "SSLHS": "squats",
    "STCR": "calf_raise",
}

CANONICAL_LABELS = [
    "squats",
    "lunges",
    "bicep_curls",
    "situps",
    "pushups",
    "tricep_extensions",
    "rows",
    "jumping_jacks",
    "shoulder_press",
    "lateral_raises",
    "pullups",
    "bench_press",
    "wrist_curl",
    "hamstring_curl",
    "leg_extension",
    "calf_raise",
    "rear_delt",
    "face_pull",
    "hip_thrust",
]

TRAIN_WORKOUTS = {"w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"}
VAL_WORKOUTS = {"w14", "w15", "w19"}
UNSEEN_TEST_WORKOUTS = {"w00", "w05", "w12", "w13", "w20"}

WHALES_COLUMNS = [
    "secondsElapsed",
    "wristMotion_rotationRateX",
    "wristMotion_rotationRateY",
    "wristMotion_rotationRateZ",
    "wristMotion_accelerationX",
    "wristMotion_accelerationY",
    "wristMotion_accelerationZ",
]


@dataclass
class Sample:
    sequence: np.ndarray
    length_50hz: int
    duration_s: float
    source: str
    source_file: str
    sample_id: str
    exercise_name: str
    original_exercise_name: str
    reps: int
    split: str
    classification_supervised: bool = True
    rep_supervised: bool = False
    rep_class_8to16: int = -1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowRecord:
    workout_id: str
    split: str
    start_idx: int
    label: int
    majority_fraction: float
    is_transition_window: bool
    window_start_s: float
    window_end_s: float
    window_start_frame: int
    window_end_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SetWise wrist-IMU datasets from MM-Fit, optionally with Whales."
    )
    parser.add_argument(
        "--dataset",
        choices=("mmfit", "merged"),
        default="mmfit",
        help="Dataset to prepare. Default isolates MM-Fit and omits Whales.",
    )
    parser.add_argument(
        "--view",
        choices=("sets", "windows"),
        default="sets",
        help="MM-Fit view to prepare. Use windows for paper-style classifier data.",
    )
    parser.add_argument("--drive-root", default=str(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--mmfit-dir", default=str(DEFAULT_MMFIT_DIR))
    parser.add_argument("--whales-dir", default=str(DEFAULT_WHALES_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--model-length", type=int, default=512)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=0.2)
    parser.add_argument(
        "--include-non-activity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include non_activity class for MM-Fit windowed classifier prep.",
    )
    parser.add_argument("--rep-min", type=int, default=8)
    parser.add_argument("--rep-max", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mount-drive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mount Google Drive at /content/drive before reading inputs.",
    )
    return parser.parse_args()


def apply_drive_root_defaults(args: argparse.Namespace) -> None:
    drive_root = Path(args.drive_root).expanduser()
    if args.mmfit_dir == str(DEFAULT_MMFIT_DIR):
        args.mmfit_dir = str(drive_root / "mm-fit-dataset")
    if args.whales_dir == str(DEFAULT_WHALES_DIR):
        args.whales_dir = str(drive_root / "Whales1and2_Raw_Labelled")
    if args.output_dir == str(DEFAULT_OUTPUT_DIR):
        args.output_dir = str(drive_root / "prepared")


def maybe_mount_drive(mount_drive: bool) -> None:
    if not mount_drive:
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--mount-drive is only available inside Google Colab.") from exc
    drive.mount("/content/drive")


def resolve_input_dir(preferred: str, fallback: Path, label: str) -> Path:
    preferred_path = Path(preferred).expanduser()
    if preferred_path.exists():
        return preferred_path
    fallback_path = fallback.expanduser()
    if fallback_path.exists():
        print(f"{label}: {preferred_path} not found; using local fallback {fallback_path}")
        return fallback_path
    raise FileNotFoundError(f"{label} directory not found: {preferred_path} or {fallback_path}")


def resolve_output_dir(preferred: str, drive_root: str) -> Path:
    preferred_path = Path(preferred).expanduser()
    drive_output = Path(drive_root).expanduser() / "prepared"
    if preferred_path == drive_output and not Path(drive_root).expanduser().exists():
        print(f"Output: {preferred_path} not found; using local fallback {LOCAL_OUTPUT_DIR}")
        return LOCAL_OUTPUT_DIR
    return preferred_path


def format_float_for_filename(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def split_name(workout_id: str) -> str:
    if workout_id in TRAIN_WORKOUTS:
        return "train"
    if workout_id in VAL_WORKOUTS:
        return "val"
    if workout_id in UNSEEN_TEST_WORKOUTS:
        return "test"
    return "unused"


def canonical_mmfit_action(action_name: str) -> str:
    return MMFIT_CANONICAL.get(action_name, action_name)


def canonical_whales_action(activity_code: str) -> str:
    if activity_code not in WHALES_CANONICAL:
        raise ValueError(f"Unmapped Whales activity code: {activity_code}")
    return WHALES_CANONICAL[activity_code]


def resample_channels(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape {values.shape}")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
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


def resample_by_time(values: np.ndarray, timestamps: np.ndarray, target_hz: float) -> np.ndarray:
    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("values and timestamps must have the same number of rows")
    if values.shape[0] == 0:
        raise ValueError("Cannot resample an empty sequence")

    order = np.argsort(timestamps)
    times = timestamps[order].astype(np.float64)
    sorted_values = values[order].astype(np.float32, copy=False)
    unique_times, unique_indices = np.unique(times, return_index=True)
    times = unique_times
    sorted_values = sorted_values[unique_indices]

    if times.shape[0] == 1:
        return sorted_values[:1].astype(np.float32, copy=False)

    duration_s = float(times[-1] - times[0])
    target_length = max(1, int(round(duration_s * target_hz)))
    target_times = np.linspace(times[0], times[-1], num=target_length, endpoint=True)
    resampled = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        resampled[:, col] = np.interp(target_times, times, sorted_values[:, col])
    return resampled


def center_accelerometer(sequence: np.ndarray) -> np.ndarray:
    centered = sequence.astype(np.float32, copy=True)
    if centered.shape[0] > 0:
        centered[:, :3] -= np.median(centered[:, :3], axis=0).astype(np.float32)
    return centered


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] <= 1:
        return values.astype(np.float32, copy=False)
    window = min(window, values.shape[0])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def active_crop(sequence: np.ndarray, target_hz: float) -> tuple[np.ndarray, dict[str, Any]]:
    if sequence.shape[0] < int(target_hz * 2):
        return sequence, {
            "active_crop_used": False,
            "active_start_50hz": 0,
            "active_end_50hz": int(sequence.shape[0]),
            "active_threshold": None,
        }

    acc_delta = np.diff(sequence[:, :3], axis=0, prepend=sequence[:1, :3])
    energy = np.linalg.norm(acc_delta, axis=1) + 0.5 * np.linalg.norm(sequence[:, 3:], axis=1)
    smoothed = moving_average(energy.astype(np.float32), max(3, int(round(target_hz * 0.5))))
    median = float(np.median(smoothed))
    p90 = float(np.percentile(smoothed, 90))
    threshold = median + 0.35 * max(p90 - median, 1e-6)
    active_indices = np.flatnonzero(smoothed > threshold)

    if active_indices.size == 0:
        return sequence, {
            "active_crop_used": False,
            "active_start_50hz": 0,
            "active_end_50hz": int(sequence.shape[0]),
            "active_threshold": threshold,
        }

    margin = int(round(target_hz * 1.0))
    start = max(0, int(active_indices[0]) - margin)
    end = min(sequence.shape[0], int(active_indices[-1]) + margin + 1)
    min_len = max(int(round(target_hz * 2.0)), 1)
    if end - start < min_len:
        return sequence, {
            "active_crop_used": False,
            "active_start_50hz": 0,
            "active_end_50hz": int(sequence.shape[0]),
            "active_threshold": threshold,
        }

    return sequence[start:end], {
        "active_crop_used": start > 0 or end < sequence.shape[0],
        "active_start_50hz": start,
        "active_end_50hz": end,
        "active_threshold": threshold,
    }


def extract_mmfit_set(
    acc_data: np.ndarray,
    gyr_data: np.ndarray,
    start_frame: int,
    end_frame: int,
    target_hz: float,
) -> tuple[np.ndarray, int, float]:
    frame_duration_s = max((end_frame - start_frame) / 30.0, 1.0 / target_hz)
    target_length = max(1, int(round(frame_duration_s * target_hz)))
    acc_mask = (acc_data[:, 0] >= start_frame) & (acc_data[:, 0] <= end_frame)
    gyr_mask = (gyr_data[:, 0] >= start_frame) & (gyr_data[:, 0] <= end_frame)
    acc_values = acc_data[acc_mask, 2:5]
    gyr_values = gyr_data[gyr_mask, 2:5]

    if acc_values.shape[0] == 0 or gyr_values.shape[0] == 0:
        raise ValueError(f"No MM-Fit smartwatch samples for frames {start_frame}-{end_frame}")

    acc_resampled = resample_channels(acc_values, target_length)
    gyr_resampled = resample_channels(gyr_values, target_length)
    combined = np.concatenate([acc_resampled, gyr_resampled], axis=1)
    return center_accelerometer(combined), target_length, frame_duration_s


def load_mmfit_labels(label_path: Path) -> list[tuple[int, int, int, str]]:
    labels: list[tuple[int, int, int, str]] = []
    with label_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            labels.append((int(row[0]), int(row[1]), int(row[2]), row[3]))
    return labels


def load_mmfit_samples(
    mmfit_dir: Path,
    target_hz: float,
    canonicalize_for_merge: bool = False,
) -> list[Sample]:
    samples: list[Sample] = []
    workout_dirs = sorted(path for path in mmfit_dir.iterdir() if path.is_dir())
    if not workout_dirs:
        raise RuntimeError(f"No MM-Fit workout directories found in {mmfit_dir}")

    for workout in workout_dirs:
        workout_id = workout.name
        acc_path = workout / f"{workout_id}_sw_l_acc.npy"
        gyr_path = workout / f"{workout_id}_sw_l_gyr.npy"
        label_path = workout / f"{workout_id}_labels.csv"
        if not acc_path.exists() or not gyr_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing MM-Fit files for {workout_id}")

        acc_data = np.load(acc_path)
        gyr_data = np.load(gyr_path)
        for set_index, (start_frame, end_frame, reps, action_name) in enumerate(
            load_mmfit_labels(label_path)
        ):
            if action_name not in MMFIT_ACTIONS:
                continue
            sequence, length_50hz, duration_s = extract_mmfit_set(
                acc_data=acc_data,
                gyr_data=gyr_data,
                start_frame=start_frame,
                end_frame=end_frame,
                target_hz=target_hz,
            )
            exercise_name = canonical_mmfit_action(action_name) if canonicalize_for_merge else action_name
            samples.append(
                Sample(
                    sequence=sequence,
                    length_50hz=length_50hz,
                    duration_s=duration_s,
                    source="mmfit",
                    source_file=str(label_path),
                    sample_id=f"mmfit:{workout_id}:set{set_index:03d}",
                    exercise_name=exercise_name,
                    original_exercise_name=action_name,
                    reps=int(reps),
                    split=split_name(workout_id),
                    classification_supervised=True,
                    rep_supervised=False,
                    rep_class_8to16=-1,
                    extra={
                        "workout_id": workout_id,
                        "set_index": set_index,
                        "frame_start": int(start_frame),
                        "frame_end": int(end_frame),
                    },
                )
            )
    return samples


def unique_sorted_time_values(
    timestamps_ms: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(timestamps_ms)
    sorted_times = timestamps_ms[order].astype(np.float64)
    sorted_values = values[order].astype(np.float32, copy=False)
    unique_times, unique_indices = np.unique(sorted_times, return_index=True)
    return unique_times, sorted_values[unique_indices]


def resample_mmfit_workout(
    acc_data: np.ndarray,
    gyr_data: np.ndarray,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_ms = float(min(acc_data[:, 1].min(), gyr_data[:, 1].min()))
    end_ms = float(max(acc_data[:, 1].max(), gyr_data[:, 1].max()))
    duration_s = max((end_ms - start_ms) / 1000.0, 1.0 / target_hz)
    target_length = max(1, int(math.floor(duration_s * target_hz)) + 1)
    target_times_ms = start_ms + (np.arange(target_length, dtype=np.float64) / target_hz) * 1000.0

    acc_times, acc_values = unique_sorted_time_values(acc_data[:, 1], acc_data[:, 2:5])
    gyr_times, gyr_values = unique_sorted_time_values(gyr_data[:, 1], gyr_data[:, 2:5])
    frame_times, frame_values = unique_sorted_time_values(acc_data[:, 1], acc_data[:, 0:1])

    acc_resampled = np.empty((target_length, 3), dtype=np.float32)
    gyr_resampled = np.empty((target_length, 3), dtype=np.float32)
    for col in range(3):
        acc_resampled[:, col] = np.interp(target_times_ms, acc_times, acc_values[:, col])
        gyr_resampled[:, col] = np.interp(target_times_ms, gyr_times, gyr_values[:, col])

    frames = np.interp(target_times_ms, frame_times, frame_values[:, 0]).astype(np.float32)
    sequence = np.concatenate([acc_resampled, gyr_resampled], axis=1)
    return sequence, frames


def label_mmfit_timesteps(
    frames: np.ndarray,
    label_rows: list[tuple[int, int, int, str]],
    include_non_activity: bool,
) -> np.ndarray:
    non_activity_id = len(MMFIT_ACTIONS)
    labels = np.full(
        frames.shape[0],
        non_activity_id if include_non_activity else -1,
        dtype=np.int64,
    )
    for start_frame, end_frame, _reps, action_name in label_rows:
        if action_name not in MMFIT_ACTION_TO_INDEX:
            continue
        mask = (frames >= start_frame) & (frames <= end_frame)
        labels[mask] = MMFIT_ACTION_TO_INDEX[action_name]
    return labels


def iter_window_records_for_workout(
    workout_id: str,
    labels: np.ndarray,
    frames: np.ndarray,
    split: str,
    target_hz: float,
    window_samples: int,
    stride_samples: int,
    num_classes: int,
    include_non_activity: bool,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    if labels.shape[0] < window_samples:
        return records

    for start_idx in range(0, labels.shape[0] - window_samples + 1, stride_samples):
        end_idx = start_idx + window_samples
        window_labels = labels[start_idx:end_idx]
        if not include_non_activity and np.any(window_labels < 0):
            valid = window_labels[window_labels >= 0]
            if valid.size == 0:
                continue
            counts = np.bincount(valid, minlength=num_classes)
            denominator = valid.size
        else:
            counts = np.bincount(window_labels, minlength=num_classes)
            denominator = window_samples

        label = int(np.argmax(counts))
        majority_count = int(counts[label])
        majority_fraction = float(majority_count / denominator)
        records.append(
            WindowRecord(
                workout_id=workout_id,
                split=split,
                start_idx=start_idx,
                label=label,
                majority_fraction=majority_fraction,
                is_transition_window=majority_fraction < 1.0,
                window_start_s=float(start_idx / target_hz),
                window_end_s=float(end_idx / target_hz),
                window_start_frame=int(round(float(frames[start_idx]))),
                window_end_frame=int(round(float(frames[end_idx - 1]))),
            )
        )
    return records


def centered_window(sequence: np.ndarray, start_idx: int, window_samples: int) -> np.ndarray:
    window = sequence[start_idx : start_idx + window_samples].astype(np.float32, copy=True)
    window[:, :3] -= np.median(window[:, :3], axis=0).astype(np.float32)
    return window


def collect_window_stats(
    mmfit_dir: Path,
    records_by_workout: dict[str, list[WindowRecord]],
    target_hz: float,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    channel_sumsq = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    count = 0

    for workout_id, records in records_by_workout.items():
        train_records = [record for record in records if record.split == "train"]
        if not train_records:
            continue
        workout = mmfit_dir / workout_id
        acc_data = np.load(workout / f"{workout_id}_sw_l_acc.npy")
        gyr_data = np.load(workout / f"{workout_id}_sw_l_gyr.npy")
        sequence, _frames = resample_mmfit_workout(acc_data, gyr_data, target_hz)
        for record in train_records:
            window = centered_window(sequence, record.start_idx, window_samples)
            channel_sum += window.sum(axis=0, dtype=np.float64)
            channel_sumsq += np.square(window, dtype=np.float64).sum(axis=0)
            count += window.shape[0]

    if count == 0:
        raise RuntimeError("Cannot fit window scaler: no train windows found")
    mean = (channel_sum / count).astype(np.float32)
    variance = (channel_sumsq / count) - np.square(channel_sum / count)
    std = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def compute_class_weights(y: np.ndarray, split: np.ndarray, num_classes: int) -> np.ndarray:
    train_y = y[split == "train"]
    counts = np.bincount(train_y, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if np.any(present):
        weights[present] = float(train_y.size) / (float(np.sum(present)) * counts[present])
    return weights


def calibrate_normalized_window_file(
    x_path: Path,
    split: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    x_memmap = np.load(x_path, mmap_mode="r+")
    channel_sum = np.zeros(x_memmap.shape[-1], dtype=np.float64)
    channel_sumsq = np.zeros(x_memmap.shape[-1], dtype=np.float64)
    count = 0

    for start in range(0, x_memmap.shape[0], chunk_size):
        end = min(start + chunk_size, x_memmap.shape[0])
        chunk_mask = split[start:end] == "train"
        if not np.any(chunk_mask):
            continue
        values = np.asarray(x_memmap[start:end][chunk_mask]).reshape(-1, x_memmap.shape[-1])
        channel_sum += values.sum(axis=0, dtype=np.float64)
        channel_sumsq += np.square(values, dtype=np.float64).sum(axis=0)
        count += values.shape[0]

    if count == 0:
        raise RuntimeError("Cannot calibrate window scaler: no train windows found")
    normalized_mean = (channel_sum / count).astype(np.float32)
    normalized_var = (channel_sumsq / count) - np.square(channel_sum / count)
    normalized_std = np.sqrt(np.maximum(normalized_var, 1e-12)).astype(np.float32)
    normalized_std = np.where(normalized_std < 1e-6, 1.0, normalized_std).astype(np.float32)

    for start in range(0, x_memmap.shape[0], chunk_size):
        end = min(start + chunk_size, x_memmap.shape[0])
        corrected = (
            (np.asarray(x_memmap[start:end]) - normalized_mean.reshape(1, 1, -1))
            / normalized_std.reshape(1, 1, -1)
        ).astype(np.float32)
        x_memmap[start:end] = corrected

    x_memmap.flush()
    del x_memmap

    effective_mean = (scaler_mean + scaler_std * normalized_mean).astype(np.float32)
    effective_std = (scaler_std * normalized_std).astype(np.float32)
    return effective_mean, effective_std


def prepare_mmfit_windows(
    mmfit_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.dataset != "mmfit":
        raise ValueError("--view windows is only supported for --dataset mmfit")

    target_hz = float(args.target_hz)
    window_samples = int(round(args.window_seconds * target_hz))
    stride_samples = int(round(args.stride_seconds * target_hz))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window-seconds and stride-seconds must produce positive sample counts")

    label_names = MMFIT_ACTIONS + (["non_activity"] if args.include_non_activity else [])
    num_classes = len(label_names)
    records: list[WindowRecord] = []
    records_by_workout: dict[str, list[WindowRecord]] = {}
    workout_sample_counts: dict[str, int] = {}

    workout_dirs = sorted(path for path in mmfit_dir.iterdir() if path.is_dir())
    if not workout_dirs:
        raise RuntimeError(f"No MM-Fit workout directories found in {mmfit_dir}")

    for workout in workout_dirs:
        workout_id = workout.name
        acc_path = workout / f"{workout_id}_sw_l_acc.npy"
        gyr_path = workout / f"{workout_id}_sw_l_gyr.npy"
        label_path = workout / f"{workout_id}_labels.csv"
        if not acc_path.exists() or not gyr_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing MM-Fit files for {workout_id}")

        acc_data = np.load(acc_path)
        gyr_data = np.load(gyr_path)
        sequence, frames = resample_mmfit_workout(acc_data, gyr_data, target_hz)
        labels = label_mmfit_timesteps(
            frames,
            load_mmfit_labels(label_path),
            include_non_activity=args.include_non_activity,
        )
        workout_records = iter_window_records_for_workout(
            workout_id=workout_id,
            labels=labels,
            frames=frames,
            split=split_name(workout_id),
            target_hz=target_hz,
            window_samples=window_samples,
            stride_samples=stride_samples,
            num_classes=num_classes,
            include_non_activity=args.include_non_activity,
        )
        records_by_workout[workout_id] = workout_records
        workout_sample_counts[workout_id] = int(sequence.shape[0])
        records.extend(workout_records)

    if not records:
        raise RuntimeError("No MM-Fit classifier windows were generated.")

    scaler_mean, scaler_std = collect_window_stats(
        mmfit_dir=mmfit_dir,
        records_by_workout=records_by_workout,
        target_hz=target_hz,
        window_samples=window_samples,
    )

    output_stem = (
        f"setwise_mmfit_windows_{int(target_hz)}hz_"
        f"{format_float_for_filename(args.window_seconds)}s_"
        f"stride{format_float_for_filename(args.stride_seconds)}"
    )
    x_path = output_dir / f"{output_stem}_X.npy"
    labels_path = output_dir / f"{output_stem}_labels.npz"
    metadata_path = output_dir / f"{output_stem}_metadata.json"

    num_windows = len(records)
    x_memmap = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_windows, window_samples, len(FEATURE_NAMES)),
    )
    y = np.empty(num_windows, dtype=np.int64)
    split_arr = np.empty(num_windows, dtype="<U8")
    workout_id_arr = np.empty(num_windows, dtype="<U8")
    window_start_s = np.empty(num_windows, dtype=np.float32)
    window_end_s = np.empty(num_windows, dtype=np.float32)
    window_start_frame = np.empty(num_windows, dtype=np.int64)
    window_end_frame = np.empty(num_windows, dtype=np.int64)
    majority_fraction = np.empty(num_windows, dtype=np.float32)
    is_transition_window = np.empty(num_windows, dtype=np.bool_)

    write_idx = 0
    for workout_id, workout_records in records_by_workout.items():
        if not workout_records:
            continue
        workout = mmfit_dir / workout_id
        acc_data = np.load(workout / f"{workout_id}_sw_l_acc.npy")
        gyr_data = np.load(workout / f"{workout_id}_sw_l_gyr.npy")
        sequence, _frames = resample_mmfit_workout(acc_data, gyr_data, target_hz)
        for record in workout_records:
            window = centered_window(sequence, record.start_idx, window_samples)
            x_memmap[write_idx] = (
                (window - scaler_mean.reshape(1, -1)) / scaler_std.reshape(1, -1)
            ).astype(np.float32)
            y[write_idx] = record.label
            split_arr[write_idx] = record.split
            workout_id_arr[write_idx] = record.workout_id
            window_start_s[write_idx] = record.window_start_s
            window_end_s[write_idx] = record.window_end_s
            window_start_frame[write_idx] = record.window_start_frame
            window_end_frame[write_idx] = record.window_end_frame
            majority_fraction[write_idx] = record.majority_fraction
            is_transition_window[write_idx] = record.is_transition_window
            write_idx += 1

    x_memmap.flush()
    del x_memmap

    scaler_mean, scaler_std = calibrate_normalized_window_file(
        x_path=x_path,
        split=split_arr,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )

    class_weight_train = compute_class_weights(y, split_arr, num_classes)
    sample_weight = class_weight_train[y].astype(np.float32)

    np.savez_compressed(
        labels_path,
        y=y,
        split=split_arr,
        workout_id=workout_id_arr,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        window_start_frame=window_start_frame,
        window_end_frame=window_end_frame,
        majority_fraction=majority_fraction,
        is_transition_window=is_transition_window,
        sample_weight=sample_weight,
        class_weight_train=class_weight_train,
        label_names=np.asarray(label_names),
        feature_names=np.asarray(FEATURE_NAMES),
        target_hz=np.asarray(target_hz, dtype=np.float32),
        window_samples=np.asarray(window_samples, dtype=np.int64),
        stride_samples=np.asarray(stride_samples, dtype=np.int64),
        window_seconds=np.asarray(args.window_seconds, dtype=np.float32),
        stride_seconds=np.asarray(args.stride_seconds, dtype=np.float32),
        scaler_mean=scaler_mean.astype(np.float32),
        scaler_std=scaler_std.astype(np.float32),
    )

    split_counts = Counter(split_arr.tolist())
    label_counts = Counter(int(label) for label in y.tolist())
    label_counts_by_split: dict[str, Counter[Any]] = defaultdict(Counter)
    for split_value, label_value in zip(split_arr.tolist(), y.tolist()):
        label_counts_by_split[split_value][int(label_value)] += 1

    metadata = {
        "dataset": "mmfit",
        "view": "windows",
        "created_by": "prepare.py",
        "x_path": str(x_path),
        "labels_path": str(labels_path),
        "target_hz": target_hz,
        "window_seconds": float(args.window_seconds),
        "stride_seconds": float(args.stride_seconds),
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "tensor_shape": [num_windows, window_samples, len(FEATURE_NAMES)],
        "feature_names": FEATURE_NAMES,
        "label_names": label_names,
        "non_activity_label": "non_activity" if args.include_non_activity else None,
        "input_paths": {"mmfit_dir": str(mmfit_dir), "whales_dir": None},
        "split_policy": "split_repo_unseen; w09/w10/w11 retained as unused",
        "split_counts": counter_to_dict(split_counts),
        "label_counts": counter_to_dict(label_counts),
        "label_counts_by_split": nested_counter_to_dict(label_counts_by_split),
        "workout_window_counts": {
            workout_id: int(len(records_by_workout[workout_id]))
            for workout_id in sorted(records_by_workout)
        },
        "workout_sample_counts_50hz": {
            workout_id: int(workout_sample_counts[workout_id])
            for workout_id in sorted(workout_sample_counts)
        },
        "normalization": {
            "accelerometer_bias": "per-window median removed from acc_x/acc_y/acc_z",
            "scaler_fit": "channel mean/std fit on train-split windows only",
            "mean": [float(value) for value in scaler_mean],
            "std": [float(value) for value in scaler_std],
            "leakage_guard": "No validation/test/unused windows are used to fit the scaler.",
        },
        "class_weight_train": [float(value) for value in class_weight_train],
        "notes": [
            "Each sample is a 5-second overlapping left-watch MM-Fit window.",
            "Window labels are assigned by majority timestep label.",
            "non_activity covers timesteps outside labeled exercise sets.",
            "Whales is omitted from this v1 classifier dataset.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"Wrote windows: {x_path}")
    print(f"Wrote labels: {labels_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Windows: {num_windows}")
    print(f"Tensor shape: {(num_windows, window_samples, len(FEATURE_NAMES))}")
    print(f"Split counts: {counter_to_dict(split_counts)}")
    print(f"Label counts: {counter_to_dict(label_counts)}")


def stable_hash_score(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    stripped = value.strip()
    if not stripped:
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return math.nan


def parse_whales_filename(path: Path) -> dict[str, Any]:
    prefix = path.stem.split("-")[0]
    tokens = prefix.split("_")
    if len(tokens) < 4:
        raise ValueError(f"Unexpected Whales filename format: {path.name}")

    activity = tokens[1]
    weight = math.nan
    set_number: int | None = None
    reps: int | None = None

    weight_idx = next((idx for idx, token in enumerate(tokens) if token.startswith("W")), None)
    set_idx = next((idx for idx, token in enumerate(tokens) if token.startswith("S")), None)
    if weight_idx is not None:
        weight_parts = [tokens[weight_idx][1:]]
        stop_idx = set_idx if set_idx is not None else len(tokens)
        weight_parts.extend(tokens[weight_idx + 1 : stop_idx])
        weight_text = ".".join(part for part in weight_parts if part)
        weight = parse_float(weight_text)

    if set_idx is not None:
        set_match = re.match(r"S(\d+)", tokens[set_idx])
        if set_match:
            set_number = int(set_match.group(1))

    rep_token = next((token for token in tokens if token.startswith("R")), None)
    if rep_token is not None:
        rep_value = parse_float(rep_token[1:])
        if math.isfinite(rep_value):
            reps = int(round(rep_value))
    elif set_idx is not None:
        for token in tokens[set_idx + 1 :]:
            rep_value = parse_float(token)
            if math.isfinite(rep_value):
                reps = int(round(rep_value))
                break

    return {
        "activity": activity,
        "weight": weight,
        "set": set_number,
        "reps": reps,
    }


def read_whales_csv(path: Path) -> dict[str, Any]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        missing = [column for column in WHALES_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows in {path}")

    times: list[float] = []
    acc_values: list[list[float]] = []
    gyr_values: list[list[float]] = []
    reps_values: list[int] = []
    activity_values: list[str] = []
    weight_values: list[float] = []
    set_values: list[int] = []

    for row in rows:
        times.append(parse_float(row.get("secondsElapsed")))
        gyr_values.append(
            [
                parse_float(row.get("wristMotion_rotationRateX")),
                parse_float(row.get("wristMotion_rotationRateY")),
                parse_float(row.get("wristMotion_rotationRateZ")),
            ]
        )
        acc_values.append(
            [
                parse_float(row.get("wristMotion_accelerationX")),
                parse_float(row.get("wristMotion_accelerationY")),
                parse_float(row.get("wristMotion_accelerationZ")),
            ]
        )

        reps_value = parse_float(row.get("reps"))
        if math.isfinite(reps_value):
            reps_values.append(int(round(reps_value)))
        activity = (row.get("activity") or "").strip()
        if activity:
            activity_values.append(activity)
        weight_value = parse_float(row.get("weight"))
        if math.isfinite(weight_value):
            weight_values.append(weight_value)
        set_value = parse_float(row.get("set"))
        if math.isfinite(set_value):
            set_values.append(int(round(set_value)))

    filename_info = parse_whales_filename(path)
    reps = reps_values[0] if reps_values else filename_info["reps"]
    if reps is None:
        raise ValueError(f"Could not determine reps for {path.name}")
    activity = activity_values[0] if activity_values else filename_info["activity"]
    weight = weight_values[0] if weight_values else filename_info["weight"]
    set_number = set_values[0] if set_values else filename_info["set"]

    times_arr = np.asarray(times, dtype=np.float64)
    acc_arr = np.asarray(acc_values, dtype=np.float32) * G_TO_M_PER_S2
    gyr_arr = np.asarray(gyr_values, dtype=np.float32)
    imu_arr = np.concatenate([acc_arr, gyr_arr], axis=1)

    return {
        "times": times_arr,
        "imu": imu_arr,
        "reps": int(reps),
        "activity": activity,
        "weight": weight,
        "set": set_number,
        "filename_reps": filename_info["reps"],
    }


def whales_general_split(source_file: str, seed: int) -> str:
    bucket = stable_hash_score(source_file, seed) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def assign_whales_splits(samples: list[Sample], rep_min: int, rep_max: int, seed: int) -> None:
    by_rep: dict[int, list[Sample]] = defaultdict(list)
    outside_range: list[Sample] = []
    for sample in samples:
        if rep_min <= sample.reps <= rep_max:
            by_rep[sample.reps].append(sample)
        else:
            outside_range.append(sample)

    for rep, group in by_rep.items():
        ordered = sorted(group, key=lambda item: stable_hash_score(item.source_file, seed))
        n_items = len(ordered)
        if n_items == 1:
            split_plan = ["train"]
        elif n_items == 2:
            split_plan = ["train", "test"]
        elif n_items == 3:
            split_plan = ["train", "train", "test"]
        elif n_items <= 5:
            split_plan = ["train"] * (n_items - 2) + ["val", "test"]
        else:
            test_count = max(1, int(round(n_items * 0.15)))
            val_count = max(1, int(round(n_items * 0.15)))
            train_count = max(1, n_items - val_count - test_count)
            split_plan = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
            split_plan = split_plan[:n_items]

        for sample, split in zip(ordered, split_plan):
            sample.split = split

    for sample in outside_range:
        sample.split = whales_general_split(sample.source_file, seed)


def load_whales_samples(
    whales_dir: Path,
    target_hz: float,
    rep_min: int,
    rep_max: int,
    seed: int,
) -> list[Sample]:
    csv_paths = sorted(whales_dir.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No Whales CSV files found in {whales_dir}")

    samples: list[Sample] = []
    for path in csv_paths:
        data = read_whales_csv(path)
        times = data["times"]
        imu = data["imu"]
        if times.size == 0:
            raise ValueError(f"No timestamps in {path.name}")

        raw_duration_s = float(np.nanmax(times) - np.nanmin(times))
        trim_start = float(np.nanmin(times) + 1.5)
        trim_end = float(np.nanmax(times) - 1.5)
        valid_mask = (
            np.isfinite(times)
            & np.all(np.isfinite(imu), axis=1)
            & (times >= trim_start)
            & (times <= trim_end)
        )
        if int(np.sum(valid_mask)) < 2:
            raise ValueError(f"Not enough valid IMU rows after trim in {path.name}")

        trimmed_times = times[valid_mask]
        trimmed_imu = imu[valid_mask]
        resampled = resample_by_time(trimmed_imu, trimmed_times, target_hz)
        cropped, crop_info = active_crop(resampled, target_hz)
        sequence = center_accelerometer(cropped)
        reps = int(data["reps"])
        activity_code = str(data["activity"])
        canonical_name = canonical_whales_action(activity_code)
        rep_supervised = rep_min <= reps <= rep_max

        samples.append(
            Sample(
                sequence=sequence,
                length_50hz=int(sequence.shape[0]),
                duration_s=float(sequence.shape[0] / target_hz),
                source="whales",
                source_file=str(path),
                sample_id=f"whales:{path.name}",
                exercise_name=canonical_name,
                original_exercise_name=activity_code,
                reps=reps,
                split="train",
                classification_supervised=True,
                rep_supervised=rep_supervised,
                rep_class_8to16=(reps - rep_min if rep_supervised else -1),
                extra={
                    "weight": none_if_nan(data["weight"]),
                    "set": data["set"],
                    "raw_duration_s": raw_duration_s,
                    "trimmed_valid_duration_s": float(trimmed_times[-1] - trimmed_times[0]),
                    "valid_rows_after_trim": int(np.sum(valid_mask)),
                    "filename_reps": data["filename_reps"],
                    **crop_info,
                },
            )
        )

    assign_whales_splits(samples, rep_min=rep_min, rep_max=rep_max, seed=seed)
    return samples


def none_if_nan(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fit_train_scaler(x_unscaled: np.ndarray, samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    train_indices = [idx for idx, sample in enumerate(samples) if sample.split == "train"]
    if not train_indices:
        raise RuntimeError("Cannot fit scaler: no train samples found")
    train_values = x_unscaled[np.asarray(train_indices)].reshape(-1, x_unscaled.shape[-1])
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize_and_stack(
    samples: list[Sample],
    model_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_unscaled = np.empty((len(samples), model_length, len(FEATURE_NAMES)), dtype=np.float32)
    for idx, sample in enumerate(samples):
        x_unscaled[idx] = resample_channels(sample.sequence, model_length)

    mean, std = fit_train_scaler(x_unscaled, samples)
    x_scaled = ((x_unscaled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    return x_scaled, mean, std


def counter_sort_key(item: Any) -> tuple[int, float | str]:
    if isinstance(item, (int, np.integer)):
        return (0, float(item))
    item_text = str(item)
    if re.fullmatch(r"-?\d+", item_text):
        return (0, float(item_text))
    return (1, item_text)


def counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=counter_sort_key)}


def nested_counter_to_dict(counter: dict[str, Counter[Any]]) -> dict[str, dict[str, int]]:
    return {key: counter_to_dict(value) for key, value in sorted(counter.items())}


def build_metadata(
    samples: list[Sample],
    x_shape: tuple[int, ...],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    args: argparse.Namespace,
    mmfit_dir: Path,
    whales_dir: Path | None,
    npz_path: Path,
    label_names: list[str],
) -> dict[str, Any]:
    source_counts = Counter(sample.source for sample in samples)
    split_counts = Counter(sample.split for sample in samples)
    source_split_counts: dict[str, Counter[Any]] = defaultdict(Counter)
    exercise_counts = Counter(sample.exercise_name for sample in samples)
    rep_counts_by_source: dict[str, Counter[Any]] = defaultdict(Counter)
    rep_supervised_counts = Counter()
    whales_activity_counts = Counter()
    whales_rep_counts = Counter()
    mmfit_rep_counts = Counter()
    mmfit_split_counts = Counter()

    for sample in samples:
        source_split_counts[sample.source][sample.split] += 1
        rep_counts_by_source[sample.source][sample.reps] += 1
        if sample.rep_supervised:
            rep_supervised_counts[sample.reps] += 1
        if sample.source == "whales":
            whales_activity_counts[sample.original_exercise_name] += 1
            whales_rep_counts[sample.reps] += 1
        if sample.source == "mmfit":
            mmfit_rep_counts[sample.reps] += 1
            mmfit_split_counts[sample.split] += 1

    train_sample_count = sum(1 for sample in samples if sample.split == "train")

    return {
        "dataset_path": str(npz_path),
        "created_by": "prepare.py",
        "dataset": args.dataset,
        "target_hz": float(args.target_hz),
        "model_length": int(args.model_length),
        "fixed_length_strategy": "temporal_resample_each_cropped_set_to_model_length",
        "rep_range": {"min": int(args.rep_min), "max": int(args.rep_max)},
        "seed": int(args.seed),
        "input_paths": {
            "mmfit_dir": str(mmfit_dir),
            "whales_dir": None if whales_dir is None else str(whales_dir),
        },
        "tensor_shape": list(x_shape),
        "feature_names": FEATURE_NAMES,
        "label_names": label_names,
        "source_counts": counter_to_dict(source_counts),
        "split_counts": counter_to_dict(split_counts),
        "source_split_counts": nested_counter_to_dict(source_split_counts),
        "exercise_counts": counter_to_dict(exercise_counts),
        "rep_counts_by_source": nested_counter_to_dict(rep_counts_by_source),
        "rep_supervised_count": int(sum(rep_supervised_counts.values())),
        "rep_supervised_counts": counter_to_dict(rep_supervised_counts),
        "classification_supervised_count": int(
            sum(1 for sample in samples if sample.classification_supervised)
        ),
        "mmfit": {
            "samples": int(source_counts["mmfit"]),
            "split_policy": "split_repo_unseen; w09/w10/w11 retained as unused",
            "split_counts": counter_to_dict(mmfit_split_counts),
            "rep_counts": counter_to_dict(mmfit_rep_counts),
            "canonical_mapping": MMFIT_CANONICAL,
        },
        "whales": {
            "samples": int(source_counts["whales"]),
            "activity_counts": counter_to_dict(whales_activity_counts),
            "rep_counts": counter_to_dict(whales_rep_counts),
            "canonical_mapping": WHALES_CANONICAL,
            "split_policy": (
                "Per-rep deterministic train/test for classes with at least two examples; "
                "validation added for classes with at least four examples. Files outside "
                "the supervised rep range use a deterministic 70/15/15 hash split."
            ),
            "preprocessing": [
                "Trim first and last 1.5 seconds.",
                "Drop rows with invalid acc/gyr channels.",
                "Convert Apple user acceleration from g to m/s^2.",
                "Resample from 100 Hz to target_hz.",
                "Apply deterministic motion-energy active cropping.",
            ],
        },
        "normalization": {
            "accelerometer_bias": "per-sequence median removed from acc_x/acc_y/acc_z",
            "scaler_fit": "channel mean/std fit on train split after fixed-length resampling",
            "scaler_train_samples": int(train_sample_count),
            "scaler_train_timesteps": int(train_sample_count * args.model_length),
            "mean": [float(value) for value in scaler_mean],
            "std": [float(value) for value in scaler_std],
            "leakage_guard": "No validation/test/unused samples are used to fit the scaler.",
        },
        "notes": [
            "MM-Fit is the default isolated dataset for v1 development.",
            "MM-Fit samples provide classification/segmentation supervision.",
            "MM-Fit rep labels are retained for paper-style counter evaluation but are not recommended as standalone learned rep-regression supervision.",
            "Whales is omitted unless --dataset merged is explicitly selected.",
            "duration_s and lengths_50hz describe the 50 Hz set signal before the final 512-step temporal resampling.",
        ],
    }


def build_arrays(
    samples: list[Sample],
    label_to_id: dict[str, int],
    label_names: list[str],
) -> dict[str, np.ndarray]:
    return {
        "lengths_50hz": np.asarray([sample.length_50hz for sample in samples], dtype=np.int64),
        "duration_s": np.asarray([sample.duration_s for sample in samples], dtype=np.float32),
        "raw_duration_s": np.asarray(
            [float(sample.extra.get("raw_duration_s", sample.duration_s)) for sample in samples],
            dtype=np.float32,
        ),
        "trimmed_valid_duration_s": np.asarray(
            [
                float(sample.extra.get("trimmed_valid_duration_s", sample.duration_s))
                for sample in samples
            ],
            dtype=np.float32,
        ),
        "source": np.asarray([sample.source for sample in samples]),
        "source_file": np.asarray([sample.source_file for sample in samples]),
        "sample_id": np.asarray([sample.sample_id for sample in samples]),
        "exercise_id": np.asarray(
            [label_to_id[sample.exercise_name] for sample in samples], dtype=np.int64
        ),
        "exercise_name": np.asarray([sample.exercise_name for sample in samples]),
        "original_exercise_name": np.asarray([sample.original_exercise_name for sample in samples]),
        "reps": np.asarray([sample.reps for sample in samples], dtype=np.int64),
        "rep_class_8to16": np.asarray([sample.rep_class_8to16 for sample in samples], dtype=np.int64),
        "rep_supervised": np.asarray([sample.rep_supervised for sample in samples], dtype=np.bool_),
        "classification_supervised": np.asarray(
            [sample.classification_supervised for sample in samples], dtype=np.bool_
        ),
        "split": np.asarray([sample.split for sample in samples]),
        "workout_id": np.asarray([str(sample.extra.get("workout_id", "")) for sample in samples]),
        "set_index": np.asarray(
            [
                int(sample.extra.get("set_index", sample.extra.get("set", -1)) or -1)
                for sample in samples
            ],
            dtype=np.int64,
        ),
        "frame_start": np.asarray(
            [int(sample.extra.get("frame_start", -1)) for sample in samples], dtype=np.int64
        ),
        "frame_end": np.asarray(
            [int(sample.extra.get("frame_end", -1)) for sample in samples], dtype=np.int64
        ),
        "weight": np.asarray(
            [
                math.nan if sample.extra.get("weight") is None else float(sample.extra.get("weight"))
                for sample in samples
            ],
            dtype=np.float32,
        ),
        "active_crop_used": np.asarray(
            [bool(sample.extra.get("active_crop_used", False)) for sample in samples],
            dtype=np.bool_,
        ),
        "active_start_50hz": np.asarray(
            [int(sample.extra.get("active_start_50hz", 0)) for sample in samples], dtype=np.int64
        ),
        "active_end_50hz": np.asarray(
            [int(sample.extra.get("active_end_50hz", sample.length_50hz)) for sample in samples],
            dtype=np.int64,
        ),
        "filename_reps": np.asarray(
            [
                -1 if sample.extra.get("filename_reps") is None else int(sample.extra.get("filename_reps"))
                for sample in samples
            ],
            dtype=np.int64,
        ),
        "feature_names": np.asarray(FEATURE_NAMES),
        "label_names": np.asarray(label_names),
    }


def print_summary(samples: list[Sample], x_shape: tuple[int, ...], metadata: dict[str, Any]) -> None:
    print(f"Samples: {len(samples)}")
    print(f"Tensor shape: {x_shape}")
    print(f"Source counts: {metadata['source_counts']}")
    print(f"Split counts: {metadata['split_counts']}")
    print(f"Rep-supervised count: {metadata['rep_supervised_count']}")
    print(f"Rep-supervised counts: {metadata['rep_supervised_counts']}")


def main() -> None:
    args = parse_args()
    apply_drive_root_defaults(args)
    maybe_mount_drive(args.mount_drive)

    mmfit_dir = resolve_input_dir(args.mmfit_dir, LOCAL_MMFIT_DIR, "MM-Fit")
    whales_dir = None
    output_dir = resolve_output_dir(args.output_dir, args.drive_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MM-Fit from {mmfit_dir}")

    if args.view == "windows":
        prepare_mmfit_windows(mmfit_dir=mmfit_dir, output_dir=output_dir, args=args)
        return

    mmfit_samples = load_mmfit_samples(
        mmfit_dir,
        target_hz=args.target_hz,
        canonicalize_for_merge=args.dataset == "merged",
    )
    print(f"Loaded MM-Fit samples: {len(mmfit_samples)}")

    if args.dataset == "merged":
        whales_dir = resolve_input_dir(args.whales_dir, LOCAL_WHALES_DIR, "Whales")
        print(f"Loading Whales from {whales_dir}")
        whales_samples = load_whales_samples(
            whales_dir,
            target_hz=args.target_hz,
            rep_min=args.rep_min,
            rep_max=args.rep_max,
            seed=args.seed,
        )
        print(f"Loaded Whales samples: {len(whales_samples)}")
        samples = mmfit_samples + whales_samples
        label_names = CANONICAL_LABELS
        output_stem = (
            f"setwise_mmfit_whales_{args.rep_min}to{args.rep_max}_"
            f"{int(args.target_hz)}hz_{args.model_length}"
        )
    else:
        samples = mmfit_samples
        label_names = MMFIT_ACTIONS
        output_stem = f"setwise_mmfit_only_{int(args.target_hz)}hz_{args.model_length}"

    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    missing_labels = sorted({sample.exercise_name for sample in samples} - set(label_to_id))
    if missing_labels:
        raise RuntimeError(f"Missing canonical label IDs: {missing_labels}")

    x_scaled, scaler_mean, scaler_std = normalize_and_stack(samples, model_length=args.model_length)
    npz_path = output_dir / f"{output_stem}.npz"
    metadata_path = output_dir / f"{output_stem}_metadata.json"

    arrays = build_arrays(samples, label_to_id, label_names)
    np.savez_compressed(
        npz_path,
        X=x_scaled,
        **arrays,
        target_hz=np.asarray(args.target_hz, dtype=np.float32),
        model_length=np.asarray(args.model_length, dtype=np.int64),
        rep_min=np.asarray(args.rep_min, dtype=np.int64),
        rep_max=np.asarray(args.rep_max, dtype=np.int64),
        scaler_mean=scaler_mean.astype(np.float32),
        scaler_std=scaler_std.astype(np.float32),
    )

    metadata = build_metadata(
        samples=samples,
        x_shape=tuple(x_scaled.shape),
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        args=args,
        mmfit_dir=mmfit_dir,
        whales_dir=whales_dir,
        npz_path=npz_path,
        label_names=label_names,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"Wrote dataset: {npz_path}")
    print(f"Wrote metadata: {metadata_path}")
    print_summary(samples, tuple(x_scaled.shape), metadata)


if __name__ == "__main__":
    main()
