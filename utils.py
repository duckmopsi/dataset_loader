import numpy as np

def get_percentile(arr, p):
    arr = np.sort(arr)
    l = arr.shape[0]
    rank = p*(l-1)
    lower_idx = int(np.floor(rank))
    higher_idx = int(np.ceil(rank))
  
    if lower_idx == higher_idx:
        return arr[lower_idx]
    else:
        lower_val = arr[lower_idx]
        higher_val = arr[higher_idx]
        return lower_val + (higher_val - lower_val) * (rank - lower_idx)
  
def eucl_dist(x, y):
    x_ = y[0]-x[0]
    y_ = y[1]-x[1]
    return np.sqrt(x_*x_ + y_*y_)