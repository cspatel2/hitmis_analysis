#%%
import numpy as np
from collections.abc import Iterable



def find_nearest(array:Iterable, targetval: float) ->tuple[int,float]:
    """finds the index and value in an array nearest to the target value.

    Args:
        array (Iterable): array to search.
        targetval (float): target value.

    Returns:
        tuple[int,float]: idex and value. Returns Iterables for both if there is more than one idx.
    """    
    dif = np.abs(np.array(array)-targetval)
    idx = np.argmin(dif)
    return idx, array[idx]