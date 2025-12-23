import numpy as np
from .io import load_json
from .transforms import interpolate_gesture, strip_timestamps, resample_stroke, get_velocity_rep, normalize_data, pad_data, resample_data, remove_first_dimension
from .utils import get_percentile, eucl_dist

class Dataset:
    def __init__(self, gestures, classes, has_timestamps, representation, interpolated=False, dt=None, classes_oh=False, class_dims=None):
        self.gestures = gestures
        self.has_timestamps = has_timestamps
        self.class_dims = class_dims
        self.dt = dt
        self.representation = representation
        self.interpolated = interpolated

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
    def from_json(cls, path, dt=0.02, drop_timestamps=False, classes_oh=False, class_dims=None):
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

            if drop_timestamps and has_timestamps:
                gesture = strip_timestamps(gesture)
                has_timestamps = False
            
            gestures.append(gesture)
            classes.append(cls_vals)

        return cls(gestures=gestures, classes=classes, has_timestamps=has_timestamps, representation="position", dt=dt, classes_oh=classes_oh, class_dims=class_dims)
    
    def __len__(self):
        return len(self.gestures)
    
    def num_classes(self):
        return self.classes.shape[1]
    
    def get_class(self, idx, ohe=False):
        """idx = class index"""
        if ohe:
            return self.classes_oh[:, idx]
        else:
            return self.classes[:, idx]
    
    def get_gestures(self):
        return self.gestures
    
    def filter_by_class(self, class_idx, value):
        mask = self.classes[:, class_idx] == value

        gestures = [g for g, m in zip(self.gestures, mask) if m]
        classes = self.classes[mask]

        return Dataset(gestures=gestures, classes=classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=True, dt=self.dt)
    
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
    
    def extract_features(self):
        length, time, start_x, start_y, end_x, end_y, area, start_v_x, start_v_y, end_v_x, end_v_y, min_v_x, min_v_y, max_v_x, max_v_y, v_25_x, v_25_y, v_50_x, v_50_y, mean_v_x, mean_v_y, v_75_x, v_75_y = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] 
        features = []
        for g in self.gestures:
            feature = []
            v_profile_x = []
            v_profile_y = []
            l = 0
            t = 0
            np_g = np.asarray(g[0])
            start_x.append(np_g[0][0])
            feature.append(np_g[0][0])
            start_y.append(np_g[0][1])
            feature.append(np_g[0][1])
            end_x.append(np_g[-1][0])
            feature.append(np_g[-1][0])
            end_y.append(np_g[-1][1])
            feature.append(np_g[-1][1])
            for i in range(np_g.shape[0]):
                if i < np_g.shape[0]-1:
                    v_profile_x.append(np.abs(np_g[i+1][0]-np_g[i][0]))
                    v_profile_y.append(np.abs(np_g[i+1][1]-np_g[i][1]))
                    l += eucl_dist(np_g[i][:2], np_g[i+1][:2])
                if self.has_timestamps:
                    t += np_g[i][2]
                else:
                    t += self.dt
            length.append(l)
            feature.append(l)
            time.append(t)
            feature.append(t)
            start_v_x.append(v_profile_x[0])
            feature.append(v_profile_x[0])
            start_v_y.append(v_profile_y[0])
            feature.append(v_profile_y[0])
            end_v_x.append(v_profile_x[-1])
            feature.append(v_profile_x[-1])
            end_v_y.append(v_profile_y[-1])
            feature.append(v_profile_y[-1])
            min_v_x.append(get_percentile(v_profile_x, 0.0))
            feature.append(get_percentile(v_profile_x, 0.0))
            min_v_y.append(get_percentile(v_profile_y, 0.0))
            feature.append(get_percentile(v_profile_y, 0.0))
            max_v_x.append(get_percentile(v_profile_x, 1.0))
            feature.append(get_percentile(v_profile_x, 1.0))
            max_v_y.append(get_percentile(v_profile_y, 1.0))
            feature.append(get_percentile(v_profile_y, 1.0))
            v_25_x.append(get_percentile(v_profile_x, 0.25))
            feature.append(get_percentile(v_profile_x, 0.25))
            v_25_y.append(get_percentile(v_profile_y, 0.25))
            feature.append(get_percentile(v_profile_y, 0.25))
            v_50_x.append(get_percentile(v_profile_x, 0.50))
            feature.append(get_percentile(v_profile_x, 0.50))
            v_50_y.append(get_percentile(v_profile_y, 0.50))
            feature.append(get_percentile(v_profile_y, 0.50))
            v_75_x.append(get_percentile(v_profile_x, 0.75))
            feature.append(get_percentile(v_profile_x, 0.75))
            v_75_y.append(get_percentile(v_profile_y, 0.75))
            feature.append(get_percentile(v_profile_y, 0.75))
            mean_v_x.append(np.mean(v_profile_x))
            feature.append(np.mean(v_profile_x))
            mean_v_y.append(np.mean(v_profile_y))
            feature.append(np.mean(v_profile_y))
            area.append((np.max(np_g[:,0])-np.min(np_g[:,0]))*(np.max(np_g[:,1])-np.min(np_g[:,1])))
            feature.append((np.max(np_g[:,0])-np.min(np_g[:,0]))*(np.max(np_g[:,1])-np.min(np_g[:,1])))
            features.append(feature)
        #return length, time, start_x, start_y, end_x, end_y, area, start_v_x, start_v_y, end_v_x, end_v_y, min_v_x, min_v_y, max_v_x, max_v_y, v_25_x, v_25_y, v_50_x, v_50_y, mean_v_x, mean_v_y, v_75_x, v_75_y      
        return features 

    def normalize_gestures(self, d_min, d_max, i_min, i_max):
        normalized = normalize_data(self.gestures, d_min, d_max, i_min, i_max, self.representation)

        return Dataset(gestures=normalized, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=True, dt=self.dt)

    def pad_gestures(self, num_points=64, value=0):
        padded = pad_data(self.gestures, num_points, rep=self.representation, value=value)

        return Dataset(gestures=padded, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=True, dt=self.dt)

    def interpolate_gestures(self, dt=0.02):
        if self.representation == "velocity":
            raise ValueError("Interpolation for velocity rep not implemented.")
        if not self.has_timestamps:
            raise ValueError("Interpolation needs timestamps.")
        interp = []
        for gesture in self.gestures:
            interp.append(interpolate_gesture(gesture, dt))
        
        return Dataset(gestures=interp, classes=self.classes, has_timestamps=False, representation=self.representation, interpolated=True, dt=dt)

    def resample_gestures(self, num_points=64):
        resampled = resample_data(self.gestures, num_points)

        return Dataset(gestures=resampled, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=True, dt=self.dt)

    def remove_first_dimension(self):
        removed = remove_first_dimension(self.gestures)

        return Dataset(gestures=removed, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=self.interpolated, dt=self.dt)

    def to_velocity(self, dt=0.02):
        if self.representation == "velocity":
            return self
        
        vel_gestures = get_velocity_rep(self.gestures, self.interpolated, dt)
        return Dataset(gestures=vel_gestures, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=self.interpolated, dt=dt)