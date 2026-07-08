import numpy as np
from .io_utils import load_json
from .transforms import interpolate_gesture, strip_timestamps, resample_stroke, get_velocity_rep, normalize_data, pad_data, unpad_data, resample_data, integrate_velocity
from .utils import get_percentile, eucl_dist

class Dataset:
    def __init__(self, gestures, classes, has_timestamps, representation, padded=False, resampled=False, interpolated=False, pos_normalized=False, velo_normalized=False, dt=None, classes_oh=False, class_dims=None, condition_types=None):
        self.gestures = gestures
        self.has_timestamps = has_timestamps
        self.class_dims = class_dims
        self.condition_types = condition_types
        self.dt = dt
        self.representation = representation
        self.interpolated = interpolated
        self.padded = padded
        self.resampled = resampled
        self.pos_normalized = pos_normalized
        self.velo_normalized = velo_normalized

        if classes_oh:
            self.classes_oh = classes
            self.classes = np.asarray([[int(np.argmax(c)) if self.condition_types is None or self.condition_types[i] == "categorical" else c[0] for i, c in enumerate(row)] for row in self.classes_oh])
        else:
            self.classes = np.asarray(classes)
            if class_dims is None:
                self.classes_oh = None
            else:
                if condition_types is None:
                    condition_types = ["categorical"] * len(class_dims)

                if len(condition_types) != len(class_dims):
                    raise ValueError("condition_types and class_dims must have same length.")
                
                self.condition_types = condition_types
                self.classes_oh = []
                for sample in self.classes:
                    sample_oh = []
                    for cls_val, dim, cond_type in zip(sample, class_dims, condition_types):
                        if cond_type == "categorical":
                            vec = np.zeros(dim, dtype=int)
                            vec[cls_val] = 1.0
                            sample_oh.append(vec)

                        elif cond_type == "continuous":
                            sample_oh.append(np.asarray([cls_val], dtype=np.float32))

                        else:
                            raise ValueError("Unknown condition type: {cond_type}")
                        
                    self.classes_oh.append(sample_oh)

    @classmethod
    def from_json(cls, path, dt=None, drop_timestamps=False, classes_oh=False, class_dims=None, condition_types=None, min_size=None, max_size=None, min_strokes=None, max_strokes=None):
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

            num_strokes = len(gesture)

            if min_strokes is not None and num_strokes < min_strokes:
                continue

            if max_strokes is not None and num_strokes > max_strokes:
                continue

            invalid = False

            for stroke in gesture:
                stroke_len = len(stroke)

                if min_size is not None and stroke_len < min_size:
                    invalid = True
                    break
                if max_size is not None and stroke_len > max_size:
                    invalid = True
                    break
            
            if invalid:
                continue
        
            if has_timestamps is None:
                has_timestamps = len(gesture[0][0]) == 3

            if drop_timestamps and has_timestamps:
                gesture = strip_timestamps(gesture)
                has_timestamps = False
            
            gestures.append(gesture)
            classes.append(cls_vals)

        return cls(gestures=gestures, classes=classes, has_timestamps=has_timestamps, representation="position", dt=dt, classes_oh=classes_oh, class_dims=class_dims, condition_types=condition_types, padded=False, resampled=False, interpolated=False, pos_normalized=False, velo_normalized=False)
    
    def __len__(self):
        return len(self.gestures)
    
    def get_config(self):
        return {"padded": self.padded, "interpolated": self.interpolated, "resampled": self.resampled, "pos_normalized": self.pos_normalized, "velo_normalized": self.velo_normalized, "dt": self.dt, "representation": self.representation}

    def num_classes(self):
        return self.classes.shape[1]
    
    def get_class(self, idx, ohe=False, class_dim=None):
        if ohe:
            if self.classes_oh is not None:
                return [sample[idx] for sample in self.classes_oh]

            if self.condition_types is not None and self.condition_types[idx] == "continuous":
                return self.classes[:, idx:idx+1]

            classes_oh = []
            for sample in self.classes:
                vec = np.zeros(class_dim, dtype=np.float32)
                vec[int(sample[idx])] = 1.0
                classes_oh.append(vec)

            return classes_oh

        return self.classes[:, idx]
    
    def get_gestures(self):
        return self.gestures
    
    def get_conditions(self, ohe=False, flatten=True):
        if ohe and self.classes_oh is not None:
            if flatten:
                return np.asarray([np.concatenate(sample, axis=-1) for sample in self.classes_oh])
            else:
                return np.asarray(self.classes_oh)
        return self.classes
    
    def filter_by_class(self, class_idx, value):
        mask = self.classes[:, class_idx] == value

        gestures = [g for g, m in zip(self.gestures, mask) if m]
        classes = self.classes[mask]

        return Dataset(gestures=gestures, classes=classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=self.interpolated, dt=self.dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=self.padded, resampled=self.resampled, pos_normalized=self.pos_normalized, velo_normalized=self.velo_normalized)
    
    def filter(self, class_filters=None, keep_condition_indices=None):
        gestures = self.gestures
        classes = np.asarray(self.classes)

        if class_filters is not None:
            mask = np.ones(len(classes), dtype=bool)

            for class_idx, allowed_values in class_filters.items():
                if self.condition_types is not None and self.condition_types[class_idx] != "categorical":
                    raise ValueError("class_filters can only be applied to categorical conditions.")

                mask &= np.isin(classes[:, class_idx], allowed_values)

            gestures = [g for g, m in zip(gestures, mask) if m]
            classes = classes[mask]

        class_dims = self.class_dims
        condition_types = self.condition_types

        if keep_condition_indices is not None:
            classes = classes[:, keep_condition_indices]

            if class_dims is not None:
                class_dims = [class_dims[i] for i in keep_condition_indices]

            if condition_types is not None:
                condition_types = [condition_types[i] for i in keep_condition_indices]

        return Dataset(gestures=gestures,classes=classes,has_timestamps=self.has_timestamps,representation=self.representation,interpolated=self.interpolated,dt=self.dt,class_dims=class_dims,condition_types=condition_types,padded=self.padded,resampled=self.resampled,pos_normalized=self.pos_normalized,velo_normalized=self.velo_normalized)

    def mean_gesture(self, mode="time", num_points=64, plot=False, save_path=None, bounds_x=None, bounds_y=None):
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

            if bounds_x is not None:
                plt.xlim(-bounds_x, bounds_x)
            if bounds_y is not None:
                plt.ylim(-bounds_y, bounds_y)
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
            if self.has_timestamps:
                t = np_g[-1][2]
            for i in range(np_g.shape[0]):
                if i < np_g.shape[0]-1:
                    if self.has_timestamps:
                        v_profile_x.append(np.abs(np_g[i+1][0]-np_g[i][0])/(np_g[i+1][2] - np_g[i][2]))
                        v_profile_y.append(np.abs(np_g[i+1][1]-np_g[i][1])/(np_g[i+1][2] - np_g[i][2]))
                    else:
                        v_profile_x.append(np.abs(np_g[i+1][0]-np_g[i][0])/self.dt)
                        v_profile_y.append(np.abs(np_g[i+1][1]-np_g[i][1])/self.dt)
                    l += eucl_dist(np_g[i][:2], np_g[i+1][:2])
                if not self.has_timestamps:
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

        if self.representation == "position":
            pos_normalized = True
            velo_normalized = False
        else:
            pos_normalized = self.pos_normalized
            velo_normalized = True

        return Dataset(gestures=normalized, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=self.interpolated, dt=self.dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=self.padded, resampled=self.resampled, pos_normalized=pos_normalized, velo_normalized=velo_normalized)

    def pad_gestures(self, num_points=64, value=0):
        padded = pad_data(self.gestures, num_points, value=value)

        return Dataset(gestures=padded, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=self.interpolated, dt=self.dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=True, resampled=self.resampled, pos_normalized=self.pos_normalized, velo_normalized=self.velo_normalized)

    def unpad_gestures(self, pad_value=-1.0):
        unpadded = unpad_data(self.gestures, pad_value)

        return Dataset(gestures=unpadded, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=self.interpolated, dt=self.dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=False, resampled=self.resampled, pos_normalized=self.pos_normalized, velo_normalized=self.velo_normalized)

    def interpolate_gestures(self, dt=0.02):
        if self.padded:
            raise ValueError("Cannot interpolate padded data.")
        if self.representation == "velocity":
            raise ValueError("Interpolation for velocity rep not implemented.")
        if not self.has_timestamps:
            raise ValueError("Interpolation needs timestamps.")
        interp = []
        for gesture in self.gestures:
            interp.append(interpolate_gesture(gesture, dt))
        
        return Dataset(gestures=interp, classes=self.classes, has_timestamps=False, representation=self.representation, interpolated=True, dt=dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=False, resampled=False, pos_normalized=self.pos_normalized, velo_normalized=self.velo_normalized)

    def resample_gestures(self, num_points=64):
        resampled = resample_data(self.gestures, num_points)

        return Dataset(gestures=resampled, classes=self.classes, has_timestamps=self.has_timestamps, representation=self.representation, interpolated=False, dt=self.dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=False, resampled=True, pos_normalized=self.pos_normalized, velo_normalized=self.velo_normalized)

    def to_velocity(self, dt=0.02):
        if self.representation == "velocity":
            return self
        
        vel_gestures = get_velocity_rep(self.gestures, self.interpolated, dt)
        return Dataset(gestures=vel_gestures, classes=self.classes, has_timestamps=self.has_timestamps, representation="velocity", interpolated=self.interpolated, dt=dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=self.padded, resampled=self.resampled, pos_normalized=self.pos_normalized, velo_normalized=False)
    
    def to_position(self, dt=None):
        if self.representation == "position":
            return self
        
        if self.velo_normalized:
            raise ValueError("Denormalize velocity first.")
        
        pos_gestures = integrate_velocity(self.gestures, dt)
        return Dataset(gestures=pos_gestures, classes=self.classes, has_timestamps=self.has_timestamps, representation="position", interpolated=self.interpolated, dt=dt, class_dims=self.class_dims, condition_types=self.condition_types, padded=self.padded, resampled=self.resampled, pos_normalized=self.pos_normalized, velo_normalized=False)