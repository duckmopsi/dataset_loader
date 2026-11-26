
import argparse
import json
import sys

from collections import defaultdict
from pathlib import Path

import numpy as np


# Allow execution from the repository root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset_loader.io_utils import load_json
from dataset_loader.transforms import interpolate_gesture
from dataset_loader.dataset import Dataset


def get_gesture_length(gesture):
    """Return the total number of points across all strokes."""
    return sum(len(stroke) for stroke in gesture)


def compute_statistics(lengths):
    """Compute descriptive statistics for gesture lengths."""
    lengths = np.asarray(lengths, dtype=np.int64)

    if len(lengths) == 0:
        return None

    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }


def get_class_names(metadata):
    """Read label names from the supported categorical metadata mapping."""
    if not isinstance(metadata, dict):
        return {}

    mappings = metadata.get("categorical_mappings", {})
    label_mapping = mappings.get("label", {})

    return {
        str(class_id): class_name
        for class_name, class_id in label_mapping.items()
    }


def print_statistics(title, stats):
    print(f"\n{title}")
    print("-" * len(title))

    if stats is None:
        print("No gestures.")
        return

    print(f"Count:  {stats['count']}")
    print(f"Min:    {stats['min']}")
    print(f"Max:    {stats['max']}")
    print(f"Mean:   {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Std:    {stats['std']:.2f}")
    print(f"P95:    {stats['p95']:.2f}")
    print(f"P99:    {stats['p99']:.2f}")


def get_remaining_timestamp_conflicts(gestures):
    """Count duplicate timestamp groups that still exist within strokes."""
    groups = 0
    affected_strokes = 0
    affected_gestures = 0
    short_strokes = 0

    for gesture in gestures:
        gesture_affected = False

        for stroke in gesture:
            points = np.asarray(stroke, dtype=float)

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Expected timestamped (x, y, t) strokes.")

            if len(points) < 2 or len(np.unique(points[:, 2])) < 2:
                short_strokes += 1

            diffs = np.diff(points[:, 2])
            duplicate_steps = diffs == 0

            if np.any(duplicate_steps):
                affected_strokes += 1
                gesture_affected = True

                groups += int(np.count_nonzero(
                    duplicate_steps & np.r_[True, ~duplicate_steps[:-1]]
                ))

        if gesture_affected:
            affected_gestures += 1

    return {
        "groups": groups,
        "affected_strokes": affected_strokes,
        "affected_gestures": affected_gestures,
        "strokes_with_fewer_than_2_unique_times": short_strokes,
    }


def audit_dataset(
    dataset_path,
    metadata_path=None,
    class_index=0,
    interpolate_dt=None,
    single_stroke_only=False,
    timestamp_conflict_strategy=None,
):
    """
    Audit gesture lengths after configurable timestamp cleanup only.
    """
    if interpolate_dt is not None and (
        not np.isfinite(interpolate_dt) or interpolate_dt <= 0
    ):
        raise ValueError("interpolate_dt must be finite and greater than zero.")

    # from_json always removes exact (x, y, t) duplicates.
    dataset = Dataset.from_json(
        path=dataset_path,
        timestamp_cleanup="duplicates_only",
    )

    exact_duplicates_report = dataset.timestamp_cleaning_report

    metadata = load_json(metadata_path) if metadata_path is not None else None
    class_names = get_class_names(metadata)

    # Optionally resolve remaining timestamp conflicts.
    strategy = (
        None if timestamp_conflict_strategy == "none"
        else timestamp_conflict_strategy
    )

    if strategy not in (None, "mean", "first", "last", "error"):
        raise ValueError(f"Unknown timestamp conflict strategy: {strategy}")

    conflict_report = None

    if strategy is not None:
        dataset, conflict_report = dataset.clean_timestamps(
            on_conflict=strategy,
            warn=False,
            return_report=True,
        )

    # Restrict only the audit selection, without modifying the Dataset.
    selected_indices = [
        i for i, gesture in enumerate(dataset.gestures)
        if not single_stroke_only or len(gesture) == 1
    ]

    selected_gestures = [
        dataset.gestures[i] for i in selected_indices
    ]

    remaining = get_remaining_timestamp_conflicts(selected_gestures)

    print("\nTIMESTAMP CLEANING")
    print("==================")
    print(
        "Exact duplicate points removed:",
        exact_duplicates_report["summary"].get("removed_points", 0)
    )
    print("Conflict strategy:", strategy or "none")

    if conflict_report is not None:
        print(
            "Conflict groups encountered:",
            conflict_report["summary"].get("conflicting_groups", 0)
        )
        print(
            "Additional points removed:",
            conflict_report["summary"].get("removed_points", 0)
        )

    print("Remaining groups in analyzed gestures:", remaining["groups"])
    print("Remaining affected gestures:", remaining["affected_gestures"])
    print("Remaining affected strokes:", remaining["affected_strokes"])
    print(
        "Strokes with fewer than 2 unique times:",
        remaining["strokes_with_fewer_than_2_unique_times"]
    )

    all_lengths = []
    class_lengths = defaultdict(list)

    interpolated_lengths = []
    class_interpolated_lengths = defaultdict(list)

    # Do not interpolate sequences with invalid timestamp intervals.
    can_interpolate = (
        interpolate_dt is not None
        and remaining["groups"] == 0
        and remaining["strokes_with_fewer_than_2_unique_times"] == 0
    )

    interpolation_warning = None

    if interpolate_dt is not None and not can_interpolate:
        interpolation_warning = (
            "Temporal interpolation skipped: unresolved duplicate timestamps "
            "or strokes with fewer than two unique timestamps."
        )
        print("\nWARNING:", interpolation_warning)

    # Analyze the selected gestures.
    for i in selected_indices:
        gesture = dataset.gestures[i]
        class_id = str(dataset.classes[i, class_index])

        length = get_gesture_length(gesture)

        all_lengths.append(length)
        class_lengths[class_id].append(length)

        # Optional temporal interpolation.
        if can_interpolate:
            interpolated = interpolate_gesture(
                gesture,
                interpolate_dt,
            )

            interpolated_length = get_gesture_length(interpolated)

            interpolated_lengths.append(interpolated_length)
            class_interpolated_lengths[class_id].append(
                interpolated_length
            )

    report = {
        "dataset": str(Path(dataset_path).resolve()),
        "metadata": (
            str(Path(metadata_path).resolve())
            if metadata_path is not None else None
        ),
        "class_index": class_index,
        "single_stroke_only": single_stroke_only,
        "timestamp_cleaning": {
            "exact_duplicates": exact_duplicates_report["summary"],
            "conflict_strategy": strategy,
            "conflict_resolution": (
                conflict_report["summary"]
                if conflict_report is not None else None
            ),
            "remaining": remaining,
        },
        "total": compute_statistics(all_lengths),
        "classes": {},
        "interpolated": None,
        "interpolation_warning": interpolation_warning,
    }

    print("\nDATASET AUDIT")
    print("=============")
    print(f"Dataset: {dataset_path}")
    print(f"Metadata: {metadata_path or 'None'}")
    print(f"Class index: {class_index}")
    print(f"Single-stroke only: {single_stroke_only}")
    print(f"Analyzed gestures: {len(selected_indices)}")

    print_statistics(
        "Overall gesture lengths",
        report["total"],
    )

    print("\nCLASS DISTRIBUTION")
    print("==================")

    for class_id in sorted(class_lengths, key=str):
        lengths = class_lengths[class_id]
        stats = compute_statistics(lengths)

        name = class_names.get(class_id, class_id)

        report["classes"][class_id] = {
            "name": name,
            **stats,
        }

        print(f"{name} (ID: {class_id}): {len(lengths)} gestures")

    print("\nPER-CLASS LENGTH STATISTICS")
    print("===========================")

    for class_id, info in report["classes"].items():
        print_statistics(
            f"{info['name']} (ID: {class_id})",
            info,
        )

    if metadata_path is not None and not class_names:
        print(
            "\nNote: Metadata loaded, but no categorical label mapping "
            "was found. Numerical class IDs are shown."
        )

    if can_interpolate:
        report["interpolated"] = {
            "dt": interpolate_dt,
            "total": compute_statistics(interpolated_lengths),
            "classes": {},
            "cutoffs": {},
        }

        print(f"\nINTERPOLATED LENGTHS (dt={interpolate_dt}s)")
        print("=" * 45)

        print_statistics(
            "Overall interpolated gesture lengths",
            report["interpolated"]["total"],
        )

        for class_id in sorted(class_interpolated_lengths, key=str):
            stats = compute_statistics(
                class_interpolated_lengths[class_id]
            )

            name = class_names.get(class_id, class_id)

            report["interpolated"]["classes"][class_id] = {
                "name": name,
                **stats,
            }

            print_statistics(
                f"{name} (ID: {class_id})",
                stats,
            )

        print("\nCUTOFF ANALYSIS")
        print("================")

        for cutoff in (128, 160, 192, 256):
            lengths = np.asarray(interpolated_lengths)

            excluded = int(np.sum(lengths > cutoff))
            total = len(lengths)

            report["interpolated"]["cutoffs"][str(cutoff)] = {
                "excluded": excluded,
                "total": total,
            }

            print(
                f"Cutoff {cutoff:3d}: {excluded:4d} excluded "
                f"({100 * excluded / total:.2f}%)"
                if total else
                f"Cutoff {cutoff:3d}: no gestures"
            )

            for class_id, class_values in sorted(
                class_interpolated_lengths.items(),
                key=lambda item: str(item[0]),
            ):
                values = np.asarray(class_values)
                n_excluded = int(np.sum(values > cutoff))

                name = str(class_names.get(class_id, class_id))

                print(
                    f"  {name:20s}: {n_excluded:4d} / {len(values):4d} "
                    f"({100 * n_excluded / len(values):.2f}%)"
                )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Descriptive audit of a gesture dataset."
    )

    parser.add_argument(
        "dataset",
        help="Path to the gesture JSON dataset.",
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON path.",
    )

    parser.add_argument(
        "--class-index",
        type=int,
        default=0,
        help="Condition index containing the gesture class (default: 0).",
    )

    parser.add_argument(
        "--single-stroke-only",
        action="store_true",
        help="Analyze only gestures consisting of exactly one stroke.",
    )

    parser.add_argument(
        "--interpolate-dt",
        type=float,
        default=None,
        help="Optionally interpolate gestures using the given time interval in seconds.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving the JSON report.",
    )

    parser.add_argument(
        "--timestamp-conflict-strategy",
        "--timestamp_conflict_strategy",
        choices=["none", "mean", "first", "last", "error"],
        default="none",
        help="Resolve remaining timestamp conflicts after removing exact duplicates.",
    )

    args = parser.parse_args()

    report = audit_dataset(
        dataset_path=args.dataset,
        metadata_path=args.metadata,
        class_index=args.class_index,
        interpolate_dt=args.interpolate_dt,
        single_stroke_only=args.single_stroke_only,
        timestamp_conflict_strategy=args.timestamp_conflict_strategy,
    )

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
