def build_dataset_pipeline(dataset, representation="position", mode="resample", num_points=64, 
                           dt=0.02, normalize=False, pos_bounds=None, velo_bounds=None, 
                           time_bounds=None, pad_value=-1.0, min_size=None, max_size=None):
    if normalize:
        d_min, d_max, i_min, i_max = pos_bounds
        dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
    
    if mode == "interpolate":
        dataset = dataset.interpolate_gestures(dt=dt)
        
        dataset = dataset.filter_by_size(min_size=min_size, max_size=max_size)
        
        if representation == "velocity":
            dataset = dataset.to_velocity(dt=dt)
            if normalize:
                d_min, d_max, i_min, i_max = velo_bounds
                dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
        dataset = dataset.pad_gestures(num_points=num_points, value=pad_value)
    elif mode == "resample":
        dataset = dataset.resample_gestures(num_points=num_points)
        
        if representation == "velocity":
            dataset = dataset.to_velocity(dt=dt)
            if normalize:
                d_min, d_max, i_min, i_max = velo_bounds
                dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)

        if normalize and time_bounds is not None:
            d_min, d_max, i_min, i_max = time_bounds
            dataset = dataset.normalize_timestamps(d_min, d_max, i_min, i_max)
    
    return dataset

def reverse_pipeline(dataset, pad_value=-1.0, pos_bounds=None, velo_bounds=None, 
                     time_bounds=None, to_position=True, mode="resample", 
                     denorm_velo=True, denorm_pos=True, denorm_time=True):

    if mode == "interpolate":
        dataset = dataset.unpad_gestures(pad_value=pad_value)

    if dataset.get_config()["representation"] == "velocity":
        if denorm_velo:
            i_min, i_max, d_min, d_max = velo_bounds
            dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)

        if to_position:
            ### TODO: fix this
            dataset = dataset.to_position(dataset.get_config()["dt"])

    if denorm_time and time_bounds is not None and dataset.has_timestamps:
        d_min, d_max, i_min, i_max = time_bounds
        dataset = dataset.normalize_timestamps(i_min, i_max, d_min, d_max)

    if denorm_pos:
        i_min, i_max, d_min, d_max = pos_bounds
        dataset = dataset.normalize_gestures(d_min, d_max, i_min, i_max)
    
    return dataset