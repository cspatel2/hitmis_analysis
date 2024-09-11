
from __future__ import annotations 
from collections.abc import Iterable
import os
import time
import subprocess
from datetime import datetime
import numpy as np
from tqdm import tqdm
import astropy.io.fits as fits
import pickle

def get_path(directory: str, prefix:str = None,suffix:str = None, wl:str =None )-> str:
    #build absolute path
    PATH = os.path.dirname(os.path.realpath(__file__))
    if isinstance(prefix, type(None)):
        dn = f'../{directory}/*'
    elif isinstance(prefix, str):
        dn = f'../{directory}/{prefix}*'
    else:
        raise ValueError("Prefix must be a string or None")

    datadir = os.path.join(PATH,dn)
    
    if not isinstance(suffix, type(None)):
        print('checked nonetype')
        datadir = dn + suffix
        if '.' not in suffix:
            datadir += '*'
    
    if isinstance(wl, (int,float,str)):
        wl = int(wl)
        datadir += f'{wl}.nc'
    return datadir

def get_tindex(t:Iterable,start:datetime, end:datetime) -> Iterable: return np.where((t >= start) & (t<= end))[0]

def open_fits(fn:str)-> np.array | np.array:
    """opens fits file using astropy.io.fits
    Args:
        fn (str): file path.

    Returns:
        tuple (np.array,np.array): image of shape (n,m) , list of headers.
    """       
    with fits.open(fn) as hdul:
        data = hdul[1].data
        header = hdul[1].header
    return np.array(data), np.array(header)

def load_pickle_file(fn:str):
    """
    load data from pickle file (.pkl).

    Args:
        fn (str): file path.

    Returns:
        (any): data with its original format. 
    """    
    with open(fn, 'rb') as file:
        dat = pickle.load(file)
    return dat


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