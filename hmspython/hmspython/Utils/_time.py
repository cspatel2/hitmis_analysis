from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime
from astropy.io import fits
import numpy as np
from dateutil import tz


def get_tindex(t:Iterable,start:datetime, end:datetime) -> Iterable: return np.where((t >= start) & (t<= end))[0]


    
def format_time(timestamp:str, input_format:str = '%Y%m%d_%H%M%S')-> str:
    """ Converts a time string from its original format to MM-YY-YYYY HH-MM-SS. 

    Args:
        timestamp (str): timestamp string.
        input_format (str, optional): timestamp string format. Defaults to '%Y%m%d_%H%M%S'.

    Returns:
        str: the time in format: '%Y-%m-%d %H:%M:%S'
    """    
    dt = datetime.strptime(timestamp, input_format)
    output_format = '%m-%d-%Y %H:%M:%S'
    formatted_str = dt.strftime(output_format)
    return formatted_str

def time_from_tstamp(ts:int) -> datetime:
    """convert timestamp in ms to datetime object.
    timstamp can be obtained from filename or file attribute.

    Args:
        ts (int): timestamp in s.

    Returns:
        datetime: datetime.datetime object in UTC.
    """    
    return datetime.fromtimestamp(ts*0.001,tz=tz.UTC)

def timestamp_from_fn(fn:str) -> int:
    """Get timestamp from the filename. This is used for file names with the format: hitmis_XXms_0_0_XXXXXXXXXXXXX.fit'

    Args:
        fn (str): file name with format: 'hitmis_XXms_0_0_XXXXXXXXXXXXX.fit'

    Returns:
        int: timestamp in ms
    """    
    return int(fn.rstrip('.fit').split('_')[-1])
