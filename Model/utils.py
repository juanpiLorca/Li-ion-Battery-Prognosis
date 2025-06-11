import numpy as np 


# Removes NaN values from a 1D numpy array.
def remove_nan(arr): 
    """Removes NaN values from a 1D numpy array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    return arr[~np.isnan(arr)]