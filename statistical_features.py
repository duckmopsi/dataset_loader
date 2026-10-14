"""Statistical XYT gesture feature extraction.

Drop this module next to the Dataset class. It does not modify gestures or the
historical (legacy) implementation of Dataset.extract_features().

The intended input is the *unprocessed* timestamped position Dataset loaded
from JSON, where t is in seconds. For the Prolific export, x and y are
viewport-normalized; pass viewport_indices=(8, 9) to recover CSS-pixel units.

Timestamp policy (v1, deliberately conservative):
- Consecutive identical (x, y, t) samples are ignored for derivatives only.
- Different coordinates with the same timestamp are flagged; derivative-based
  speed/acceleration for the entire gesture become NaN, not an invented value.
- Negative or nonfinite times are flagged similarly.
- Spatial measurements use the original samples (zero-length repeats do not
  change geometric distances).

These decisions can be revisited after inspecting QC reports, without changing
raw data or prior results.
"""

from __future__ import annotations

import numpy as np


# Tolerance is used only to decide if successive *CSS-pixel* positions are
# computationally identical; it is NOT an outlier threshold.
_SAME_POSITION_ATOL = 1e-7


def _nan_stats():
    return {"mean_speed": np.nan, "peak_speed": np.nan,
            "mean_acceleration": np.nan, "peak_acceleration": np.nan}


def _validate_dataset(dataset):
    if not dataset.has_timestamps or dataset.representation != "position":
        raise ValueError("Statistical extraction requires timestamped position (x, y, t) data.")
    for flag in ("padded", "resampled", "interpolated", "pos_normalized", "velo_normalized"):
        if getattr(dataset, flag, False):
            raise ValueError(
                f"Dataset.{flag}=True: use the unprocessed, timestamped JSON Dataset "
                "for statistical feature extraction."
            )


def _stroke_features(stroke, gesture_idx, stroke_idx, scales):
    """Return one-stroke features, QC counts, and derivative integrals."""
    a = np.asarray(stroke, dtype=float)
    row = {
        "gesture_idx": gesture_idx,
        "stroke_idx": stroke_idx,
        "n_points": len(stroke),
        "duration_s": np.nan,
        "path_length": np.nan,
        "displacement": np.nan,
        "bbox_width": np.nan,
        "bbox_height": np.nan,
        "bbox_area": np.nan,
        "total_turn_angle_rad": np.nan,
        "n_zero_dt_same_position": 0,
        "n_zero_dt_moving": 0,
        "n_negative_dt": 0,
        "max_zero_dt_jump": 0.0,
        "valid_spatial": False,
        "valid_temporal": False,
        **_nan_stats(),
    }
    # Internal quantities are kept separate from the public result.
    integrals = {"speed_distance": 0.0, "speed_duration": 0.0,
                 "acceleration_integral": 0.0, "acceleration_duration": 0.0}

    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2:
        return row, integrals
    if not np.all(np.isfinite(a)):
        return row, integrals

    xy = a[:, :2] * scales
    t = a[:, 2]
    displacements = np.diff(xy, axis=0)
    segment_lengths = np.linalg.norm(displacements, axis=1)
    dt = np.diff(t)
    zero = dt == 0
    negative = dt < 0
    moving = zero & (segment_lengths > _SAME_POSITION_ATOL)

    row["n_zero_dt_same_position"] = int(np.sum(zero & ~moving))
    row["n_zero_dt_moving"] = int(np.sum(moving))
    row["n_negative_dt"] = int(np.sum(negative))
    row["max_zero_dt_jump"] = float(np.max(segment_lengths[zero])) if np.any(zero) else 0.0

    row["path_length"] = float(segment_lengths.sum())
    row["displacement"] = float(np.linalg.norm(xy[-1] - xy[0]))
    span = np.ptp(xy, axis=0)
    row["bbox_width"], row["bbox_height"] = map(float, span)
    row["bbox_area"] = float(span[0] * span[1])
    row["valid_spatial"] = True

    nonzero_steps = displacements[segment_lengths > _SAME_POSITION_ATOL]
    if len(nonzero_steps) >= 2:
        orientations = np.arctan2(nonzero_steps[:, 1], nonzero_steps[:, 0])
        turning = (np.diff(orientations) + np.pi) % (2 * np.pi) - np.pi
        row["total_turn_angle_rad"] = float(np.abs(turning).sum())
    elif len(nonzero_steps) == 1:
        row["total_turn_angle_rad"] = 0.0

    if not np.any(negative):
        row["duration_s"] = float(t[-1] - t[0])

    # Different points at exactly the same time are a measurement ambiguity.
    # We do not artificially introduce tiny deltas or silently collapse them.
    if np.any(negative) or np.any(moving):
        return row, integrals

    # Remove exact same-time, same-position repeats only in local calculations.
    keep = np.r_[True, ~zero]
    xy_clean = xy[keep]
    t_clean = t[keep]
    if len(t_clean) < 2 or np.any(np.diff(t_clean) <= 0):
        return row, integrals

    dt_clean = np.diff(t_clean)
    delta_xy = np.diff(xy_clean, axis=0)
    velocity = delta_xy / dt_clean[:, None]
    speed = np.linalg.norm(velocity, axis=1)

    row["valid_temporal"] = True
    row["mean_speed"] = float(np.linalg.norm(delta_xy, axis=1).sum() / dt_clean.sum())
    row["peak_speed"] = float(np.max(speed))
    integrals["speed_distance"] = float(np.linalg.norm(delta_xy, axis=1).sum())
    integrals["speed_duration"] = float(dt_clean.sum())

    if len(velocity) >= 2:
        mids = (t_clean[:-1] + t_clean[1:]) / 2.0
        acceleration_dt = np.diff(mids)
        acceleration = np.linalg.norm(np.diff(velocity, axis=0) / acceleration_dt[:, None], axis=1)
        row["mean_acceleration"] = float(np.average(acceleration, weights=acceleration_dt))
        row["peak_acceleration"] = float(np.max(acceleration))
        integrals["acceleration_integral"] = float(np.sum(acceleration * acceleration_dt))
        integrals["acceleration_duration"] = float(np.sum(acceleration_dt))

    return row, integrals


def extract_statistical_features(dataset, viewport_indices=None):
    """Extract per-stroke and per-gesture statistical features.

    Args:
        dataset: Unprocessed, timestamped Dataset with XYT strokes.
        viewport_indices: Optional indices (width, height) into dataset.classes.
            For the Prolific metadata use (8, 9). If omitted, distances and
            derivatives stay in the input coordinates (typically viewport01).

    Returns:
        dict with `gesture_features`, `stroke_features`, `qc_summary`, `units`.
        Each gesture_features row has `gesture_idx` matching its original sample.
    """
    _validate_dataset(dataset)
    if viewport_indices is not None and (
        len(viewport_indices) != 2 or
        any(not isinstance(i, (int, np.integer)) or i < 0 for i in viewport_indices)
    ):
        raise ValueError("viewport_indices must be a pair of nonnegative condition indices.")

    stroke_rows = []
    gesture_rows = []
    qc = {"n_gestures": len(dataset.gestures),
          "n_strokes": 0, "n_strokes_zero_dt_moving": 0,
          "n_strokes_negative_dt": 0, "n_gestures_invalid_temporal": 0,
          "n_gestures_invalid_spatial": 0}

    for gi, gesture in enumerate(dataset.gestures):
        if viewport_indices is None:
            scales = np.ones(2, dtype=float)
        else:
            try:
                scales = np.asarray(dataset.classes[gi, list(viewport_indices)], dtype=float)
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"Could not read viewport dimensions for gesture {gi}.") from exc
            if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError(f"Invalid viewport dimensions for gesture {gi}: {scales!r}.")

        rows = []
        totals = []
        min_t, max_t = [], []
        for si, stroke in enumerate(gesture):
            row, integral = _stroke_features(stroke, gi, si, scales)
            rows.append(row)
            totals.append(integral)
            stroke_rows.append(row)
            qc["n_strokes"] += 1
            qc["n_strokes_zero_dt_moving"] += int(row["n_zero_dt_moving"] > 0)
            qc["n_strokes_negative_dt"] += int(row["n_negative_dt"] > 0)
            a = np.asarray(stroke, dtype=float)
            if a.ndim == 2 and a.shape[1] == 3 and len(a) >= 2 and np.all(np.isfinite(a)):
                min_t.append(float(np.min(a[:, 2])))
                max_t.append(float(np.max(a[:, 2])))

        spatial_ok = bool(rows) and all(r["valid_spatial"] for r in rows)
        temporal_ok = spatial_ok and all(r["valid_temporal"] for r in rows)
        qc["n_gestures_invalid_spatial"] += int(not spatial_ok)
        qc["n_gestures_invalid_temporal"] += int(not temporal_ok)

        out = {
            "gesture_idx": gi,
            "n_strokes": len(rows),
            "n_points": int(sum(r["n_points"] for r in rows)),
            "n_zero_dt_same_position": int(sum(r["n_zero_dt_same_position"] for r in rows)),
            "n_zero_dt_moving": int(sum(r["n_zero_dt_moving"] for r in rows)),
            "n_negative_dt": int(sum(r["n_negative_dt"] for r in rows)),
            "max_zero_dt_jump": max((r["max_zero_dt_jump"] for r in rows), default=0.0),
            "valid_spatial": spatial_ok,
            "valid_temporal": temporal_ok,
            "path_length": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
            "bbox_area": np.nan,
            "directness": np.nan,
            "total_turn_angle_rad": np.nan,
            "duration_active_s": np.nan,
            "duration_elapsed_s": np.nan,
            **_nan_stats(),
        }

        if spatial_ok:
            out["path_length"] = float(sum(r["path_length"] for r in rows))
            out["directness"] = (
                float(sum(r["displacement"] for r in rows) / out["path_length"])
                if out["path_length"] > 0 else np.nan
            )
            if all(np.isfinite(r["total_turn_angle_rad"]) for r in rows):
                out["total_turn_angle_rad"] = float(sum(r["total_turn_angle_rad"] for r in rows))
            # Gesture-level bounding box: use ALL original points, not a sum
            # of stroke bounding boxes.
            xy_parts = [np.asarray(s, dtype=float)[:, :2] * scales for s in gesture]
            xy_all = np.concatenate(xy_parts, axis=0)
            span = np.ptp(xy_all, axis=0)
            out["bbox_width"], out["bbox_height"] = map(float, span)
            out["bbox_area"] = float(span[0] * span[1])

        if rows and all(np.isfinite(r["duration_s"]) for r in rows):
            # Active duration is SUM of stroke durations; concurrent strokes
            # contribute multiple times. Elapsed duration is wall-clock span,
            # provided all strokes share a timestamp origin.
            out["duration_active_s"] = float(sum(r["duration_s"] for r in rows))
            out["duration_elapsed_s"] = float(max(max_t) - min(min_t)) if min_t else np.nan

        if temporal_ok:
            duration = sum(a["speed_duration"] for a in totals)
            distance = sum(a["speed_distance"] for a in totals)
            if duration > 0:
                out["mean_speed"] = float(distance / duration)
            out["peak_speed"] = float(max(r["peak_speed"] for r in rows))
            acc_dur = sum(a["acceleration_duration"] for a in totals)
            if acc_dur > 0 and all(np.isfinite(r["mean_acceleration"]) for r in rows):
                out["mean_acceleration"] = float(
                    sum(a["acceleration_integral"] for a in totals) / acc_dur
                )
                out["peak_acceleration"] = float(max(r["peak_acceleration"] for r in rows))

        gesture_rows.append(out)

    unit = "css_px" if viewport_indices is not None else "input_coordinate_unit"
    return {
        "gesture_features": gesture_rows,
        "stroke_features": stroke_rows,
        "qc_summary": qc,
        "units": {"distance": unit, "time": "seconds", "speed": f"{unit}/s", "acceleration": f"{unit}/s^2"},
    }
