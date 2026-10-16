import numpy as np
import copy
import warnings
from collections import Counter

from scipy.interpolate import interp1d

def interpolate_stroke(stroke, dt):
    stroke = np.asarray(stroke)
    
    x, y, t = stroke[:, 0], stroke[:, 1], stroke[:, 2]
    t = t - t[0]

    t_new = np.arange(0, t[-1]+0.00001, dt)

    fx = interp1d(t, x, kind="linear", fill_value="extrapolate")
    fy = interp1d(t, y, kind="linear", fill_value="extrapolate")

    return np.stack([fx(t_new), fy(t_new)], axis=-1).tolist()

def interpolate_gesture(gesture, dt):
    return [interpolate_stroke(stroke, dt) for stroke in gesture]

def resample_data(gestures, num_points):
    """
    Resample gestures to a fixed number of time steps.

    Args:
        gestures: Gesture sequences
        num_points: target sequence length

    Returns:
        Resampled gesture sequences
    """
    interpolated = []
    for gesture in gestures:
        new_gesture = []
        for stroke in gesture:
            if len(stroke[0]) == 3:
                x = [p[0] for p in stroke]
                y = [p[1] for p in stroke]
                t = [p[2] for p in stroke]
                T_uniform = np.linspace(t[0], t[-1], num_points)

                interp_x = interp1d(t, x, kind='linear', fill_value='extrapolate')
                interp_y = interp1d(t, y, kind='linear', fill_value='extrapolate')
                X_uniform = interp_x(T_uniform)
                Y_uniform = interp_y(T_uniform)

                new_gesture.append([[X_uniform[i], Y_uniform[i], T_uniform[i]] for i in range(len(T_uniform))])
            else:
                new_gesture.append(resample_stroke(stroke, num_points))

        interpolated.append(new_gesture)
    return interpolated

def resample_stroke(stroke, num_points):
    stroke = np.asarray(stroke)
    t_old = np.linspace(0, 1, len(stroke))
    t_new = np.linspace(0, 1, num_points)
    x = np.interp(t_new, t_old, stroke[:, 0])
    y = np.interp(t_new, t_old, stroke[:, 1])
    return np.stack([x, y], axis=1)

def strip_timestamps(gesture):
    return [[[p[0], p[1]] for p in stroke] for stroke in gesture]

def get_velocity_rep(gestures, interpolated=False, data_step=0.02):
    """
    Convert position sequences to velocity representation.
    """
    velo_rep = []
    for gesture in gestures:
        new_rep = []
        for stroke in gesture:
            initial = stroke[0]
            x = [p[0] for p in stroke]
            y = [p[1] for p in stroke]
            if len(stroke[0]) > 2:
                t = [p[2] for p in stroke]
                dt = np.diff(np.asarray(t))
            else:
                t = None
                dt = data_step

            dx = np.diff(np.asarray(x))
            dy = np.diff(np.asarray(y))

            vx = dx/dt
            vy = dy/dt

            if interpolated or t is None:
                seq = [initial]
                for i in range(len(dx)):
                    seq.append([vx[i], vy[i]])
            else:
                seq = [[initial[0], initial[1], 0]]
                for i in range(len(dx)):
                    seq.append([vx[i], vy[i], t[i + 1]])
            new_rep.append(seq)
        velo_rep.append(new_rep)
        
    return velo_rep

def integrate_velocity(gestures, dt=None):
    """
    Convert velocity representation back to position trajectories.

    Args:
        gestures: Velocity sequences
        dt: Optional fixed time step - if None, time dim has to be available

    Returns:
        Integrated position trajectories
    """    
    new_gestures = []

    for g in gestures:
        new_gesture = []
        for stroke in g:
            new_gesture.append(integrate_velocity_stroke(stroke, dt))
        new_gestures.append(new_gesture)

    return new_gestures

def integrate_velocity_stroke(stroke, dt=None):
    vx = [p[0] for p in stroke[1:]]
    vy = [p[1] for p in stroke[1:]]

    x0 = stroke[0][0]
    y0 = stroke[0][1]
    x = np.zeros(len(vx) + 1).tolist() if dt is None else [0]
    y = np.zeros(len(vy) + 1).tolist() if dt is None else [0]
    
    x[0] = x0
    y[0] = y0
    
    if dt is not None:
        x.extend(np.cumsum(vx, axis=-1)*dt + x0)
        y.extend(np.cumsum(vy, axis=-1)*dt + y0)
    
        new_stroke = np.stack([x, y], axis=-1)   
    else:
        t = [p[2] for p in stroke]
    
        for i in range(len(vx)):
            delta_t = t[i+1] - t[i]
            x[i+1] = x[i] + vx[i] * delta_t
            y[i+1] = y[i] + vy[i] * delta_t
    
        new_stroke = np.stack([x, y, t], axis=-1)
        
    return new_stroke

def normalize_data(data, d_min, d_max, i_min, i_max, rep='position'):
    """
    Normalize gesture coordinates to a target interval.

    Args:
        data: Gesture sequences
        d_min, d_max: Original data range
        i_min, i_max: Target interval
        rep: Representation type
    
    Returns:
        Normalized data copy
    """
    dd = copy.deepcopy(data)
    for i in range(len(dd)):
        strokes = dd[i]
        for stroke in strokes:
            start_idx = 0 if rep=='position' else 1
            for d in stroke[start_idx:]:
                d[0] = np.clip(d[0], a_min=d_min, a_max=d_max)
                d[1] = np.clip(d[1], a_min=d_min, a_max=d_max)
                d[0] = (d[0] - d_min) / (d_max - d_min) * (i_max - i_min) + i_min
                d[1] = (d[1] - d_min) / (d_max - d_min) * (i_max - i_min) + i_min
    return dd

def pad_data(data, length, value=0):
    """
    Pad gesture sequences to fixed length.

    Args:
        data: Gesture sequences
        length: Target sequence length
        value: Padding value

    Returns:
        Padded numpy array
    """ 
    new_gestures = []
    for gesture in data:
        new_gesture = []
        for stroke in gesture:
            new_stroke = list(stroke)
            if len(new_stroke[0]) == 3:
                raise ValueError("Padding not allowed for data with timestamps.")
            for i in range(length - len(new_stroke)):    
                new_stroke.append([value, value])
            new_gesture.append(np.asarray(new_stroke))
        new_gestures.append(np.asarray(new_gesture))
    return np.asarray(new_gestures)

def unpad_data(data, pad_value=-1.0):
    """
    Remove padding tokens from padded gesture sequences.

    Args:
        data: Padded data sequences
        pad_value: Threshold below which samples are considered padding

    Returns:
        List of unpadded data
    """
    new_data = []

    for g in data:
        new_gesture = []
        for stroke in g:
            new_gesture.append(unpad_data_stroke(stroke, pad_value))
        new_data.append(new_gesture)

    return new_data

def unpad_data_stroke(stroke, pad_value=-1.0):
    new_stroke = []

    for xy in stroke:
        if xy[0] <= pad_value or xy[1] <= pad_value:
            break
        new_stroke.append(xy)

    if len(new_stroke) < 3:
        # random token as invalid gesture
        return [[-50, -50], [-50, -50]]
    else:
        return new_stroke

def remove_first_dimension(data):
    """
    Removes the first singleton dimension from gesture arrays. (i.e. (N, 1, T, D) -> (N, T, D))
    """
    new_data = []
    for d in data:
        new_data.append(d[0])
    return np.asarray(new_data)

def add_first_dimension(data):
    """
    Adds a singleton dimension to gesture arrays. (i.e. (N, T, D) -> (N, 1, T, D))
    """
    return np.asarray([[d] for d in data])

def clean_dataset_timestamps(dataset, *, on_conflict="last", viewport_indices=None,
                             warn=True, return_report=False):
    """Return an independent Dataset with strictly increasing intra-stroke times.

    Parameters
    ----------
    dataset : Dataset
        Timestamped position Dataset, NOT velocity representation or padded.
    on_conflict : {'last', 'first', 'mean', 'error'}
        When multiple different positions have the same timestamp, select
        last / first / mean position, or reject the dataset with an error.
        The default 'last' is an explicit deterministic convention, NOT
        a recovery of the unknown true timing of these points.
    viewport_indices : tuple[int, int] | None
        Indices of viewport_width and viewport_height in dataset.classes.
        Only needed for reporting conflict jumps in CSS pixels. Coordinates
        themselves remain in the original input units.
    warn : bool
        Emit one summary warning when conflicting-position timestamps exist.
    return_report : bool
        True: return (cleaned_dataset, report). False: return cleaned_dataset.
        The report is always attached at cleaned_dataset.timestamp_cleaning_report.

    No input data are modified. All labels and Dataset configuration flags
    are retained. No automatic distance threshold or exclusion is applied.
    """

    _CONFLICT_POLICIES = {"last", "first", "mean", "error"}

    if on_conflict not in _CONFLICT_POLICIES:
        raise ValueError(f"on_conflict must be one of {sorted(_CONFLICT_POLICIES)}")
    if not dataset.has_timestamps or dataset.representation != "position":
        raise ValueError("Timestamp cleanup requires timestamped position (x, y, t) data")
    if dataset.padded:
        raise ValueError("Timestamp cleanup cannot operate on padded data")
    if viewport_indices is not None:
        if len(viewport_indices) != 2:
            raise ValueError("viewport_indices must be a pair of condition indices")
        vw_idx, vh_idx = viewport_indices
    else:
        vw_idx, vh_idx = None, None

    output_gestures = []
    totals = Counter()
    affected_strokes = []
    gesture_flags = []

    for gesture_idx, gesture in enumerate(dataset.gestures):
        new_gesture = []
        gesture_conflicts = False
        gesture_short = False
        gesture_max_jump = 0.0
        gesture_duplicate_pairs = 0
        gesture_moving_pairs = 0

        if viewport_indices is not None:
            vw = float(dataset.classes[gesture_idx, vw_idx])
            vh = float(dataset.classes[gesture_idx, vh_idx])
            if not np.isfinite(vw) or not np.isfinite(vh) or vw <= 0 or vh <= 0:
                raise ValueError(f"Invalid viewport dimensions at gesture {gesture_idx}")
            scales = np.array([vw, vh], dtype=float)
        else:
            scales = np.ones(2, dtype=float)

        for stroke_idx, stroke in enumerate(gesture):
            points = np.asarray(stroke, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Malformed XYT stroke at gesture {gesture_idx}, stroke {stroke_idx}")
            if not np.all(np.isfinite(points)):
                raise ValueError(f"Non-finite XYT data at gesture {gesture_idx}, stroke {stroke_idx}")
            if len(points) > 1 and np.any(np.diff(points[:, 2]) < 0):
                raise ValueError(f"Decreasing timestamps at gesture {gesture_idx}, stroke {stroke_idx}")

            totals["strokes"] += 1
            totals["original_points"] += len(points)
            clean_points = []
            ident_pairs = 0
            conflicting_pairs = 0
            conflicting_groups = 0
            max_jump = 0.0
            group_count = 0

            i = 0
            while i < len(points):
                j = i + 1
                while j < len(points) and points[j, 2] == points[i, 2]:
                    j += 1
                block = points[i:j]
                group_count += 1

                if len(block) > 1:
                    steps = np.diff(block[:, :2], axis=0)
                    step_norms = np.linalg.norm(steps * scales, axis=1)
                    movement = step_norms > 0
                    conflicting_pairs += int(np.count_nonzero(movement))
                    ident_pairs += int(np.count_nonzero(~movement))
                    if np.any(movement):
                        conflicting_groups += 1
                        max_jump = max(max_jump, float(step_norms.max()))
                        if on_conflict == "error":
                            raise ValueError(
                                f"Different positions at the same timestamp in "
                                f"gesture {gesture_idx}, stroke {stroke_idx}, time {points[i, 2]}"
                            )

                if on_conflict == "first":
                    representative = block[0].copy()
                elif on_conflict == "mean":
                    representative = block[0].copy()
                    representative[:2] = np.mean(block[:, :2], axis=0)
                else:  # 'last' and 'error' (when no conflicts)
                    representative = block[-1].copy()
                clean_points.append(representative.tolist())
                i = j

            new_gesture.append(clean_points)
            removed = len(points) - len(clean_points)
            totals["removed_points"] += removed
            totals["identical_pairs"] += ident_pairs
            totals["conflicting_pairs"] += conflicting_pairs
            totals["conflicting_groups"] += conflicting_groups
            totals["cleaned_points"] += len(clean_points)
            if removed:
                totals["affected_strokes"] += 1
            if conflicting_groups:
                totals["conflicted_strokes"] += 1
                gesture_conflicts = True
            if len(clean_points) < 3:
                totals["strokes_with_fewer_than_3_unique_times"] += 1
                gesture_short = True
            gesture_max_jump = max(gesture_max_jump, max_jump)
            gesture_duplicate_pairs += removed
            gesture_moving_pairs += conflicting_pairs

            if removed or len(clean_points) < 3:
                affected_strokes.append({
                    "gesture_idx": gesture_idx,
                    "stroke_idx": stroke_idx,
                    "points_before": len(points),
                    "points_after": len(clean_points),
                    "identical_pairs": ident_pairs,
                    "conflicting_pairs": conflicting_pairs,
                    "conflicting_groups": conflicting_groups,
                    "max_conflict_jump": max_jump,
                })

        output_gestures.append(new_gesture)
        gesture_flags.append({
            "gesture_idx": gesture_idx,
            "had_duplicate_timestamps": gesture_duplicate_pairs > 0,
            "had_conflicting_positions": gesture_conflicts,
            "has_short_stroke_after_cleanup": gesture_short,
            "removed_points": gesture_duplicate_pairs,
            "conflicting_pairs": gesture_moving_pairs,
            "max_conflict_jump": gesture_max_jump,
        })
        if gesture_conflicts:
            totals["conflicted_gestures"] += 1
        if gesture_duplicate_pairs:
            totals["affected_gestures"] += 1

    totals["gestures"] = len(output_gestures)
    report = {
        "method": "group_consecutive_equal_timestamps",
        "on_conflict": on_conflict,
        "jump_unit": "css_px" if viewport_indices is not None else "input_coordinate_units",
        "summary": dict(totals),
        "affected_strokes": affected_strokes,
        "gesture_flags": gesture_flags,
    }

    cleaned = type(dataset)(
        gestures=output_gestures,
        classes=np.asarray(dataset.classes).copy(),
        has_timestamps=dataset.has_timestamps,
        representation=dataset.representation,
        padded=dataset.padded,
        resampled=dataset.resampled,
        interpolated=dataset.interpolated,
        pos_normalized=dataset.pos_normalized,
        velo_normalized=dataset.velo_normalized,
        dt=dataset.dt,
        classes_oh=False,
        class_dims=copy.deepcopy(dataset.class_dims),
        condition_types=copy.deepcopy(dataset.condition_types),
    )
    # Preserve any externally supplied one-hot encoding when the constructor
    # cannot rebuild it (e.g. class_dims=None).
    if getattr(dataset, "classes_oh", None) is not None and cleaned.classes_oh is None:
        cleaned.classes_oh = copy.deepcopy(dataset.classes_oh)
    cleaned.timestamp_cleaning_report = report

    if warn and totals["conflicting_groups"]:
        warnings.warn(
            f"Timestamp cleanup found {totals['conflicting_groups']} groups with "
            f"different positions at the same time in "
            f"{totals['conflicted_gestures']} gestures. "
            f"Applied on_conflict={on_conflict!r}; review "
            "cleaned.timestamp_cleaning_report before statistical inference.",
            RuntimeWarning, stacklevel=2,
        )
    if return_report:
        return cleaned, report
    return cleaned