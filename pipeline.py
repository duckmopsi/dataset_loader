import numpy as np

def build_dataset_pipeline(dataset, representation="position", mode="resample", num_points=64, dt=0.02, normalize=False, pos_bounds=None, velo_bounds=None, pad_value=-1.0):
    if normalize:
        d_min, d_max, i_min, i_max = pos_bounds
        dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
        #print("nach normalize: ", dataset.get_gestures()[0])
    
    if mode == "interpolate":
        dataset = dataset.interpolate_gestures(dt=dt)
        #print("nach interpolate: ", dataset.get_gestures()[0])
        if representation == "velocity":
            dataset = dataset.to_velocity(dt=dt)
            #print("nach to velo: ", dataset.get_gestures()[0])
            if normalize:
                d_min, d_max, i_min, i_max = velo_bounds
                dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
                #print("nach velo norm: ", dataset.get_gestures()[0])
        dataset = dataset.pad_gestures(num_points=num_points, value=pad_value)
        #print("nach pad: ", dataset.get_gestures()[0])
    elif mode == "resample":
        dataset = dataset.resample_gestures(num_points=num_points)
        #print("nach resample: ", dataset.get_gestures()[0])
        if representation == "velocity":
            dataset = dataset.to_velocity(dt=dt)
            #print("nach to velo: ", dataset.get_gestures()[0])
            if normalize:
                d_min, d_max, i_min, i_max = velo_bounds
                dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
                #print("nach velo norm: ", dataset.get_gestures()[0])
    
    return dataset

def reverse_pipeline(dataset, pad_value=-1.0, pos_bounds=None, velo_bounds=None, to_position=True, mode="resample", denorm_velo=True, denorm_pos=True):

    if mode == "interpolate":
        dataset = dataset.unpad_gestures(pad_value=pad_value)
        #print("nach unpad: ", dataset.get_gestures()[0])
    if dataset.get_config()["representation"] == "velocity":
        if denorm_velo:
            i_min, i_max, d_min, d_max = velo_bounds
            dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
            #print("nach denorm velo: ", dataset.get_gestures()[0])
            if to_position:
                dataset = dataset.to_position(dataset.get_config()["dt"])
                #print("nach to pos: ", dataset.get_gestures()[0])
    if denorm_pos:
        i_min, i_max, d_min, d_max = pos_bounds
        dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
        #print("nach denorm pos: ", dataset.get_gestures()[0])
    
    return dataset