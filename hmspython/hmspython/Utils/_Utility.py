#%%
from __future__ import annotations
import numpy as np
from collections.abc import Iterable
import itertools



def find_nearest(array:Iterable, targetval: float) ->tuple[int,float]:
    """finds the index and value in an array nearest to the target value.

    Args:
        array (Iterable): array to search.
        targetval (float): target value.

    Returns:
        tuple[int,float]: idex and value. Returns Iterables for both if there is more than one idx.
    """    
    dif = np.abs(np.array(array)-targetval)
    idx = np.nanargmin(dif)
    return idx, array[idx]

def correct_unit_of_angle(angle:float, convert_to:str) ->float:
    """ converts angle into desired unit.

    Args:
        angle (float): angle can be in radian or degrees.
        convert_to (str): must be one of the following: 'degrees' or 'radians'.

    Raises:
        ValueError: convert_to units must be degrees or radians.

    Returns:
        float: angle in desired unit.
    """    
    if isinstance(angle, Iterable):
        return np.asarray([correct_unit_of_angle(a,convert_to) for a in angle])
    convert_to = convert_to.replace('s','')
    if convert_to not in 'radian' and convert_to not in 'degree':
        raise ValueError("convert_to units must be degrees or radians.")
    if (0<= np.abs(angle)<= 2*np.pi):
        if convert_to.lower() in "degrees":
            return np.rad2deg(angle)
        else:
            return angle
    
    elif (0<= np.abs(angle)<= 360.):
        if convert_to.lower() in "radians":
            return np.deg2rad(angle)
        else:
            return angle\

def flatten_list(listxd:list)-> list:
    """ Flattens an unevely shappend multi-demnesional list to a 1D list.

    Args:
        list2d (list): multi-demnsional list.

    Returns:
        list: Flattened list.
    """    
    return list(itertools.chain.from_iterable(listxd))
# %%
