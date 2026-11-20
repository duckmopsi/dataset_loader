
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


def get_gesture_length(gesture):
    """
    Return total number of points in a gesture.

    Supports:
        Single-stroke: [[x, y, t], ...]
        Multi-stroke:  [[[x, y, t], ...], ...]
    """
    if not isinstance(gesture, list):
        raise ValueError("Gesture must be a list.")

    if not gesture:
        return 0

    # Single-stroke
    if (isinstance(gesture[0], list) and gesture[0] and isinstance(gesture[0][0], (int, float))):
        return len(gesture)

    # Multi-stroke
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
    """
    Extract class names when metadata contains a simple
    class_names mapping or list.

    Unknown metadata formats are not interpreted.
    """
    if not isinstance(metadata, dict):
        return {}

    mappings = metadata.get("categorical_mappings", {})
    label_mapping = mappings.get("label", {})

    return {
        str(class_id): class_name for class_name, class_id in label_mapping.items()
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


def audit_dataset(dataset_path, metadata_path=None, class_index=0, interpolate_dt=None, single_stroke_only=False):
    """
    Analyze the original dataset without filtering,
    resampling, normalization or padding.
    """
    raw = load_json(dataset_path)

    if not isinstance(raw, list):
        raise ValueError("Expected a list of dataset entries.")

    metadata = (load_json(metadata_path) if metadata_path is not None else None)

    class_names = get_class_names(metadata)

    all_lengths = []
    class_lengths = defaultdict(list)

    interpolated_lengths = []
    class_interpolated_lengths = defaultdict(list)

    for i, item in enumerate(raw):
        # Expected format: [gesture, condition_0, condition_1, ...]
        if not isinstance(item, list) or len(item) <= class_index + 1:
            raise ValueError(
                f"Invalid dataset entry at index {i}: "
                "missing gesture or class information."
            )

        gesture = item[0]
        class_id = str(item[class_index + 1])

        # Normalize structure to a list of strokes.
        if not gesture:
            strokes = []
        elif isinstance(gesture[0][0], (int, float)):
            strokes = [gesture]
        else:
            strokes = gesture

        # Optional single-stroke restriction.
        if single_stroke_only and len(strokes) != 1:
            continue

        # Original number of points.
        length = get_gesture_length(gesture)

        all_lengths.append(length)
        class_lengths[class_id].append(length)

        # Optional temporal interpolation.
        if interpolate_dt is not None:
            if not strokes:
                interpolated_length = 0
            else:
                if any(
                    not stroke or len(stroke[0]) < 3
                    for stroke in strokes
                ):
                    raise ValueError(
                        "Temporal interpolation requires timestamps."
                    )

                interpolated = interpolate_gesture(
                    strokes, interpolate_dt
                )

                interpolated_length = sum(
                    len(stroke) for stroke in interpolated
                )

            interpolated_lengths.append(interpolated_length)
            class_interpolated_lengths[class_id].append(
                interpolated_length
            )

    report = {
        "dataset": str(Path(dataset_path).resolve()),
        "metadata": (str(Path(metadata_path).resolve()) if metadata_path is not None else None),
        "class_index": class_index,
        "total": compute_statistics(all_lengths),
        "classes": {},
    }

    print("\nDATASET AUDIT")
    print("=============")
    print(f"Dataset: {dataset_path}")
    print(f"Metadata: {metadata_path or 'None'}")
    print(f"Class index: {class_index}")

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
            "\nNote: Metadata loaded, but no 'class_names' "
            "mapping was found. Numerical class IDs are shown."
        )

    report["single_stroke_only"] = single_stroke_only
    report["interpolated"] = None

    if interpolate_dt is not None:
        report["interpolated"] = {
            "dt": interpolate_dt,
            "total": compute_statistics(interpolated_lengths),
            "classes": {},
        }

        print(f"\nINTERPOLATED LENGTHS (dt={interpolate_dt}s)")
        print("=" * 45)

        print_statistics(
            "Overall interpolated gesture lengths",
            report["interpolated"]["total"],
        )

        for class_id in sorted(
            class_interpolated_lengths,
            key=lambda x: int(x),
        ):
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

        for cutoff in [128, 160, 192, 256]:
            lengths = np.asarray(interpolated_lengths)

            excluded = np.sum(lengths > cutoff)

            print(
                f"Cutoff {cutoff:3d}: "
                f"{excluded:4d} excluded "
                f"({100 * excluded / len(lengths):.2f}%)"
            )

            for class_id, class_values in sorted(
                class_interpolated_lengths.items(),
                key=lambda item: int(item[0]),
            ):
                values = np.asarray(class_values)
                n_excluded = np.sum(values > cutoff)

                print(
                    f"  {class_names.get(class_id, class_id):20s}: "
                    f"{n_excluded:4d} / {len(values):4d} "
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
        help="Optionally interpolate gestures using the given "
            "time interval in seconds.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving the JSON report.",
    )

    args = parser.parse_args()

    report = audit_dataset(
        dataset_path=args.dataset,
        metadata_path=args.metadata,
        class_index=args.class_index,
        interpolate_dt=args.interpolate_dt,
        single_stroke_only=args.single_stroke_only
    )

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
