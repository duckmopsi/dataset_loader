import numpy as np
from scipy.interpolate import interp1d
import copy

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
    interpolated = []
    for gesture in gestures:
        new_gesture = []
        for stroke in gesture:
            x = [p[0] for p in stroke]
            y = [p[1] for p in stroke]
            t = [p[2] for p in stroke]
            T_uniform = np.linspace(t[0], t[-1], num_points)

            interp_x = interp1d(t, x, kind='linear', fill_value='extrapolate')
            interp_y = interp1d(t, y, kind='linear', fill_value='extrapolate')
            X_uniform = interp_x(T_uniform)
            Y_uniform = interp_y(T_uniform)

            new_gesture.append([[X_uniform[i], Y_uniform[i], T_uniform[i]] for i in range(len(T_uniform))])

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

def get_velocity_rep(gestures, interpolated=False, DATA_STEP=0.02):
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
                dt = DATA_STEP

            dx = np.diff(np.asarray(x))
            dy = np.diff(np.asarray(y))

            vx = dx/dt
            vy = dy/dt

            if not interpolated:
                seq = [initial]
                for i in range(len(dx)):
                    seq.append([vx[i], vy[i]])
            else:
                seq = [[initial[0], initial[1], 0]]
                for i in range(len(dx)):
                    seq.append([vx[i], vy[i], t[i+1]])
            new_rep.append(seq)
        velo_rep.append(new_rep)
    return velo_rep

def normalize_data(data, d_min, d_max, i_min, i_max, rep='position'):
    dd = copy.deepcopy(data)
    for i in range(len(dd)):
        strokes = dd[i]
        for gesture in strokes:
            start_idx = 0 if rep=='position' else 1
            for d in gesture[start_idx:]:
                d[0] = np.clip(d[0], a_min=d_min, a_max=d_max)
                d[1] = np.clip(d[1], a_min=d_min, a_max=d_max)
                d[0] = (d[0] - d_min) / (d_max - d_min) * (i_max - i_min) + i_min
                d[1] = (d[1] - d_min) / (d_max - d_min) * (i_max - i_min) + i_min
    return dd

def pad_data(data, length, rep='position', value=0):
    new_gestures = []
    for gesture in data:
        new_gesture = []
        for stroke in gesture:
            new_stroke = list(stroke)
            for i in range(length - len(new_stroke)):    
                new_stroke.append([value, value])
            new_gesture.append(np.asarray(new_stroke))
        new_gestures.append(np.asarray(new_gesture))
    return np.asarray(new_gestures)

def remove_first_dimension(data):
    new_data = []
    for d in data:
        new_data.append(d[0])
    return np.asarray(new_data)