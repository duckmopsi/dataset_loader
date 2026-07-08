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

def get_velocity_rep(gestures, interpolated=False, DATA_STEP=0.02):
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
                dt = DATA_STEP

            dx = np.diff(np.asarray(x))
            dy = np.diff(np.asarray(y))

            vx = dx/dt
            vy = dy/dt

            if interpolated:
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
    x = np.zeros(len(vx)+1).tolist() if not dt else [0]
    y = np.zeros(len(vy)+1).tolist() if not dt else [0]
    
    x[0] = x0
    y[0] = y0
    
    if dt:
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
        representation: Representation type
    
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
        pad_value: Value used for padding

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
        if xy[0] < pad_value or xy[1] < pad_value:
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