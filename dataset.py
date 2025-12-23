import numpy as np
from .io import load_json
from .transforms import interpolate_gesture, strip_timestamps, resample_stroke

class Dataset:
    def __init__(self, gestures, classes, has_timestamps, dt=None, classes_oh=False, class_dims=None):
        self.gestures = gestures
        self.has_timestamps = has_timestamps
        self.class_dims = class_dims
        self.dt = dt

        if classes_oh:
            self.classes_oh = classes
            self.classes = np.asarray([[int(np.argmax(c)) for c in row] for row in self.classes_oh])
        else:
            self.classes = np.asarray(classes)
            if class_dims is None:
                self.classes_oh = None
            else:
                self.classes_oh = []
                for sample in self.classes:
                    sample_oh = []
                    for cls_val, dim in zip(sample, class_dims):
                        vec = np.zeros(dim, dtype=int)
                        vec[cls_val] = 1
                        sample_oh.append(vec)
                    self.classes_oh.append(sample_oh)

    @classmethod
    def from_json(cls, path, interpolate=False, dt=0.02, drop_timestamps=False, classes_oh=False, class_dims=None):
        raw = load_json(path)

        gestures = []
        classes = []
        has_timestamps = None

        for item in raw:
            gesture = item[0]
            if len(gesture) == 0:
                continue
            cls_vals = item[1:]

            if isinstance(gesture[0][0], (int, float)):
                gesture = [gesture]

            if has_timestamps is None:
                has_timestamps = len(gesture[0][0]) == 3

            if interpolate:
                if not has_timestamps:
                    raise ValueError("Interpolation needs timestamps.")
                gesture = interpolate_gesture(gesture, dt)

            if drop_timestamps and has_timestamps:
                gesture = strip_timestamps(gesture)
                has_timestamps = False
            
            gestures.append(gesture)
            classes.append(cls_vals)

        return cls(gestures=gestures, classes=classes, has_timestamps=has_timestamps, classes_oh=classes_oh, class_dims=class_dims)
    
    def __len__(self):
        return len(self.gestures)
    
    def num_classes(self):
        return self.classes.shape[1]
    
    def get_class(self, idx):
        """idx = class index"""
        return self.classes[:, idx]
    
    def get_gestures(self):
        return self.gestures
    
    def filter_by_class(self, class_idx, value):
        mask = self.classes[:, class_idx] == value

        gestures = [g for g, m in zip(self.gestures, mask) if m]
        classes = self.classes[mask]

        return Dataset(gestures, classes, self.has_timestamps)
    
    def mean_gesture(self, mode="time", num_points=64, plot=False, save_path=None):
        gestures = self.gestures
        
        max_strokes = max(len(g) for g in gestures)
        mean_gesture = []

        for s_idx in range(max_strokes):
            strokes = [g[s_idx] for g in gestures if len(g) > s_idx]

            if len(strokes) == 0:
                continue

            if mode=='time':
                T_max = max(len(s) for s in strokes)
                all_s = np.zeros((T_max, 2))
                counts = np.zeros(T_max)

                for stroke in strokes:
                    stroke = np.asarray(stroke)
                    N = len(stroke)
                    all_s[:N] += stroke[:, :2]
                    counts[:N] += 1

                mean_stroke = all_s / counts[:, None]
            
            elif mode=='position':
                resampled = []
                for stroke in strokes:
                    resampled.append(resample_stroke(stroke, num_points))

                mean_stroke = np.mean(np.stack(resampled, axis=0), axis=0)

            else:
                raise ValueError("mode must be 'time' or 'position'")
            
            mean_gesture.append(mean_stroke)

        if plot:
            import matplotlib.pyplot as plt

            for g in gestures:
                for stroke in g:
                    s = np.asarray(stroke)
                    plt.plot(s[:, 0], s[:, 1], color="gray", alpha=0.15)

            for stroke in mean_gesture:
                plt.plot(stroke[:, 0], stroke[:, 1], linewidth=3)

            plt.gca().invert_yaxis()
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()

        return mean_gesture