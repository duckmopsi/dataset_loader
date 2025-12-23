import numpy as np
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

def resample_stroke(stroke, num_points):
    stroke = np.asarray(stroke)
    t_old = np.linspace(0, 1, len(stroke))
    t_new = np.linspace(0, 1, num_points)
    x = np.interp(t_new, t_old, stroke[:, 0])
    y = np.interp(t_new, t_old, stroke[:, 1])
    return np.stack([x, y], axis=1)

def strip_timestamps(gesture):
    return [[[p[0], p[1]] for p in stroke] for stroke in gesture]