
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
from skimage import exposure
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

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

def open_fits(fn:str,timestamp:bool=True)-> np.array | np.array:
    """opens fits file using astropy.io.fits
    Args:
        fn (str): file path.

    Returns:
        tuple (np.array,np.array): image of shape (n,m) , list of headers.
    """       
    with fits.open(fn) as hdul:
        data = hdul[1].data
        header = hdul[1].header
    if timestamp: return np.array(data) , int(hdul[1].header['HIERARCH TIMESTAMP'])
    else: return np.array(data)

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

def time_from_tstamp(ts:int) -> datetime:
    """convert timestamp in ms to datetime object.
    timstamp can be obtained from filename or file attribute.

    Args:
        ts (int): timestamp in s.

    Returns:
        _type_: datetime.datetime object
    """    
    return datetime.fromtimestamp(ts*0.001)

def timestamp_from_fn(fn:str) -> int:
    """Get timestamp from the filename. This is used for file names with the format: hitmis_XXms_0_0_XXXXXXXXXXXXX.fit'

    Args:
        fn (str): file name with format: 'hitmis_XXms_0_0_XXXXXXXXXXXXX.fit'

    Returns:
        int: timestamp in ms
    """    
    return int(fn.rstrip('.fit').split('_')[-1])

def open_eclipse_data(fn:str, crop_xpix:int = 96, crop_ypix:int = 96,imgsize:int = 3008,normalize:bool = False,save:bool= False, plot:bool=False):
    """
    Loads  data fits files from eclipse day or before, crops img to mosaic, and mirrors image so as to match the orientation of the final image as originally seen by the detector.

    Args:
        fn (str): file path
        crop_xpix (int, optional):number of cols to crop from the top. Defaults to 95.
        crop_ypix (int, optional): number of rows to crop from bottom. Defaults to 95.
        imgsize (int, optional):shape of final square image, it can be 1024x1024 or 3008x3008.. Defaults to 3008.
        normalize (bool, optional): If True, normalize the image using skimage.exposure.equalize_hist(). Defaults to False.
        save (bool, optional): If True, saves the new cropped and fliped image as a new file with prefix of the fn as "cropped". Defaults to False.
        plot (bool, optional): If True, plots the new cropped and fliped image. Defaults to False.

    Returns:
        np.array: HMS img
    """    
    img_data,_ = open_fits(fn)
 
    if normalize: img_data = exposure.equalize_hist(img_data)
    cropped_img_data = img_data[:-crop_ypix, :-crop_xpix]
    # Resize the cropped image to 3008x3008 pixels
    zoom_factor = (imgsize / cropped_img_data.shape[0], imgsize/ cropped_img_data.shape[1])
    resized_img_data = zoom(cropped_img_data, zoom_factor, order=3)  # order=3 for cubic interpolation
    # Reverse the image along both axes
    reversed_img_data = np.flip(resized_img_data, axis=(0, 1))
    
    # # Save the reversed and resized image as a new FITS file
    if save:
        hdu = fits.PrimaryHDU(reversed_img_data)
        hdul_new = fits.HDUList([hdu])
        output_fits_path = f"cropped_{fn.split('/')[-1]}"
        hdul_new.writeto(output_fits_path, overwrite=True)
    if plot:
        plt.figure()
        plt.imshow(reversed_img_data,cmap = 'viridis')
        plt.colorbar()
        plt.show()
        
    return reversed_img_data