from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SMARTWATCH_MODALITIES = (
    "sw_l_acc",
    "sw_l_gyr",
    "sw_l_hr",
    "sw_r_acc",
    "sw_r_gyr",
    "sw_r_hr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the MM-Fit smartwatch subset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="mm-fit-dataset",
        help="Path to the extracted MM-Fit dataset directory.",
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


def format_minutes(minutes: float) -> str:
    return f"{minutes:.2f}"


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    workouts = workout_dirs(dataset_dir)

    availability = Counter()
    missing = {modality: [] for modality in SMARTWATCH_MODALITIES}
    rows_total = Counter()
    duration_total = Counter()
    rate_values = defaultdict(list)
    median_dt_values = defaultdict(list)
    nonpositive_dt_total = Counter()
    gap_max_ms = defaultdict(list)
    per_workout = {}

    for workout in workouts:
        workout_stats = {}
        for modality in SMARTWATCH_MODALITIES:
            path = workout / f"{workout.name}_{modality}.npy"
            if not path.exists():
                missing[modality].append(workout.name)
                workout_stats[modality] = None
                continue

            availability[modality] += 1
            data = np.load(path)
            timestamps = data[:, 1].astype(np.float64)
            deltas = np.diff(timestamps)
            duration_s = float((timestamps[-1] - timestamps[0]) / 1000.0) if len(timestamps) > 1 else 0.0
            mean_rate_hz = float((len(timestamps) - 1) / duration_s) if duration_s > 0 else float("nan")
            median_dt_ms = float(np.median(deltas)) if len(deltas) else float("nan")

            rows_total[modality] += int(data.shape[0])
            duration_total[modality] += duration_s
            rate_values[modality].append(mean_rate_hz)
            median_dt_values[modality].append(median_dt_ms)
            nonpositive_dt_total[modality] += int(np.sum(deltas <= 0))
            gap_max_ms[modality].append(float(deltas.max()) if len(deltas) else 0.0)

            workout_stats[modality] = {
                "rows": int(data.shape[0]),
                "frame_start": int(data[0, 0]),
                "frame_end": int(data[-1, 0]),
                "duration_s": duration_s,
            }

        per_workout[workout.name] = workout_stats

    activity_sets = Counter()
    activity_reps = Counter()
    activity_duration_frames = Counter()
    per_workout_activity_minutes = {}
    watch_timeline = {}

    for workout in workouts:
        labels = load_labels(workout / f"{workout.name}_labels.csv")
        activity_frames = 0
        for start, end, reps, activity in labels:
            duration_frames = end - start
            activity_frames += duration_frames
            activity_sets[activity] += 1
            activity_reps[activity] += reps
            activity_duration_frames[activity] += duration_frames

        per_workout_activity_minutes[workout.name] = activity_frames / 30.0 / 60.0

        starts = []
        ends = []
        for modality in SMARTWATCH_MODALITIES:
            modality_stats = per_workout[workout.name][modality]
            if modality_stats is None:
                continue
            starts.append(modality_stats["frame_start"])
            ends.append(modality_stats["frame_end"])

        if starts and ends:
            span_frames = max(ends) - min(starts)
            watch_timeline[workout.name] = {
                "timeline_minutes": span_frames / 30.0 / 60.0,
                "activity_minutes": activity_frames / 30.0 / 60.0,
                "activity_fraction": activity_frames / span_frames if span_frames else float("nan"),
            }

    # Aggregate signal statistics in a second pass to keep memory use predictable.
    signal_stats = {}
    for modality in SMARTWATCH_MODALITIES:
        dims = 1 if modality.endswith("hr") else 3
        count = 0
        value_sum = np.zeros(dims, dtype=np.float64)
        value_sq_sum = np.zeros(dims, dtype=np.float64)
        value_min = np.full(dims, np.inf)
        value_max = np.full(dims, -np.inf)
        finite_ok = True
        nonmonotonic_files = []

        for workout in workouts:
            path = workout / f"{workout.name}_{modality}.npy"
            if not path.exists():
                continue

            data = np.load(path)
            values = data[:, 2:].astype(np.float64)
            timestamps = data[:, 1].astype(np.float64)
            deltas = np.diff(timestamps)
            if np.any(deltas < 0):
                nonmonotonic_files.append(workout.name)
            finite_ok &= bool(np.isfinite(data).all())

            count += values.shape[0]
            value_sum += values.sum(axis=0)
            value_sq_sum += np.square(values).sum(axis=0)
            value_min = np.minimum(value_min, values.min(axis=0))
            value_max = np.maximum(value_max, values.max(axis=0))

        if count == 0:
            continue

        mean = value_sum / count
        variance = np.maximum(value_sq_sum / count - np.square(mean), 0.0)
        signal_stats[modality] = {
            "count": count,
            "mean": mean,
            "std": np.sqrt(variance),
            "min": value_min,
            "max": value_max,
            "finite_ok": finite_ok,
            "nonmonotonic_files": nonmonotonic_files,
        }

    print("# MM-Fit Smartwatch Analysis")
    print()
    print("## Scope")
    print(f"- Workouts: {len(workouts)} ({workouts[0].name} to {workouts[-1].name})")
    print("- Modalities: left/right smartwatch accelerometer, gyroscope, and heart rate")
    print("- Raw file layout: IMU arrays are `[frame, timestamp_ms, x, y, z]`; HR arrays are `[frame, timestamp_ms, bpm]`")
    print()
    print("## Modality Coverage")
    for modality in SMARTWATCH_MODALITIES:
        median_rate = np.median(rate_values[modality]) if rate_values[modality] else float("nan")
        median_dt = np.median(median_dt_values[modality]) if median_dt_values[modality] else float("nan")
        print(
            f"- {modality}: {availability[modality]}/{len(workouts)} workouts, "
            f"{rows_total[modality]:,} rows, {duration_total[modality] / 3600.0:.2f} hours, "
            f"median file rate {median_rate:.2f} Hz, median dt {median_dt:.1f} ms"
        )
    print()
    print("## Missing Files")
    for modality in SMARTWATCH_MODALITIES:
        print(f"- {modality}: {format_list(missing[modality])}")
    print()
    print("## Label Summary")
    total_labeled_minutes = sum(per_workout_activity_minutes.values())
    print(f"- Total labeled exercise time: {format_minutes(total_labeled_minutes)} minutes")
    for activity, sets in activity_sets.most_common():
        total_minutes = activity_duration_frames[activity] / 30.0 / 60.0
        avg_seconds = (activity_duration_frames[activity] / 30.0) / sets
        avg_reps = activity_reps[activity] / sets
        print(
            f"- {activity}: {sets} sets, {activity_reps[activity]} reps, "
            f"{total_minutes:.2f} min total, {avg_seconds:.2f} sec/set, {avg_reps:.2f} reps/set"
        )
    print()
    print("## Workout-Level Coverage")
    fractions = [stats["activity_fraction"] for stats in watch_timeline.values()]
    print(
        f"- Activity fraction within smartwatch timelines: mean {np.mean(fractions):.3f}, "
        f"median {np.median(fractions):.3f}, min {np.min(fractions):.3f}, max {np.max(fractions):.3f}"
    )
    for workout_name, stats in sorted(watch_timeline.items()):
        modalities_present = [
            modality
            for modality in SMARTWATCH_MODALITIES
            if per_workout[workout_name][modality] is not None
        ]
        print(
            f"- {workout_name}: {stats['timeline_minutes']:.2f} min timeline, "
            f"{stats['activity_minutes']:.2f} min labeled activity, "
            f"{stats['activity_fraction']:.3f} active fraction, "
            f"modalities={','.join(modalities_present)}"
        )
    print()
    print("## Quality Checks")
    for modality in SMARTWATCH_MODALITIES:
        stats = signal_stats[modality]
        mean = ", ".join(f"{value:.3f}" for value in stats["mean"])
        std = ", ".join(f"{value:.3f}" for value in stats["std"])
        min_value = ", ".join(f"{value:.3f}" for value in stats["min"])
        max_value = ", ".join(f"{value:.3f}" for value in stats["max"])
        print(
            f"- {modality}: finite={stats['finite_ok']}, nonmonotonic_ts={format_list(stats['nonmonotonic_files'])}, "
            f"nonpositive_dt={nonpositive_dt_total[modality]:,}, max_gap_ms={max(gap_max_ms[modality]):.0f}, "
            f"mean=[{mean}], std=[{std}], min=[{min_value}], max=[{max_value}]"
        )


if __name__ == "__main__":
    main()
