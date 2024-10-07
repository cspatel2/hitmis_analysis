
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
from PIL import Image
from glob import glob

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

def open_eclipse_data(img:str|np.ndarray, crop_xpix:int = 96, crop_ypix:int = 96,mirror:bool=True,imgsize:int = 3008,normalize:bool = False,savefits:bool= False,savepng:bool=False, plot:bool=False):
    """
    Loads  data fits files from eclipse day or before, crops img to mosaic, and mirrors image so as to match the orientation of the final image as originally seen by the detector.

    Args:
        img (str|np.ndarray): file path, or 2D img array.
        crop_xpix (int, optional):number of cols to crop from the top. Defaults to 96.
        crop_ypix (int, optional): number of rows to crop from bottom. Defaults to 96.
        mirror (Bool, optional):if True, image will be mirrors about both axes to match the orientation at the detector. if true: img at detector, if False: img at mosaic. Defaults to True
        imgsize (int, optional):shape of final square image, it can be 1024x1024 or 3008x3008.. Defaults to 3008.
        normalize (bool, optional): If True, normalize the image using skimage.exposure.equalize_hist(). Defaults to False.
        savefits (bool, optional): If True, saves the new cropped and fliped image as a new .fits file with prefix of the fn as "cropped". Defaults to False.
        savepng (bool, optional): If True, saves the new cropped and fliped image as a new .png file with prefix of the fn as "cropped". Defaults to False.
        plot (bool, optional): If True, plots the new cropped and fliped image. Defaults to False.

    Returns:
        np.array: HMS img
    """ 
    if isinstance(img,(np.ndarray,list)): img_data = img   
    elif isinstance(img,str):
        if os.path.exists(img): img_data= open_fits(img,False)
        else: raise ValueError(f"img path '{img}' does not exist. ")
    else: raise ValueError("img must be valid file path or 2D img array.")
    if normalize: img_data = exposure.equalize_hist(img_data)
    cropped_img_data = img_data[:-crop_ypix, :-crop_xpix]
    # Resize the cropped image to mgsize pixels
    zoom_factor = (imgsize / cropped_img_data.shape[0], imgsize/ cropped_img_data.shape[1])
    resized_img_data = zoom(cropped_img_data, zoom_factor, order=3)  # order=3 for cubic interpolation
    if mirror: #image at detector
        # Reverse the image along both axes
        final_img_data = np.flip(resized_img_data, axis=(0, 1))
    else:#image at mosaic
        final_img_data = resized_img_data
    
    # # Save the reversed and resized image as a new FITS file
    if savefits and isinstance(img,str):
        hdu = fits.PrimaryHDU(final_img_data)
        hdul_new = fits.HDUList([hdu])
        dirpath,fname = os.path.split(img)
        output_fits_fname = f'cropped_{fname}'
        output_fits_path = os.path.join(dirpath,output_fits_fname)
        hdul_new.writeto(output_fits_path, overwrite=True)
    if savepng and isinstance(img,str):
        dirpath,fname = os.path.split(img)
        fname = fname.split('.')[0]
        output_png_fname = f'cropped_{fname}.png'
        output_png_path = os.path.join(dirpath,output_png_fname)
        saveimg = Image.fromarray(final_img_data)
        saveimg.save(output_png_path)
        
    if plot:
        plt.figure()
        plt.imshow(final_img_data,cmap = 'viridis')
        plt.colorbar()
        plt.show()
        
    return final_img_data

def read_metadata(metadata_file:str)-> dict:
    """converts metadata .txt file to dict. This func is used in conjuction with png_to_fits() for images taken with the ASIStudio software using the ZWO camera.

    Args:
        metadata_file (str): path of metadata .txt (or .PNG.txt) file.

    Returns:
        dict: metadata.
    """    
    metadata = {}
    with open(metadata_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Skip empty lines and the header line
            if line.strip() and not line.startswith('['):
                key, value = line.strip().split(' = ', 1)
                metadata[key] = value
    return metadata

def png_to_fits(png_file:str, fits_file:str, metadata_file:None) -> None:
    """converts .png images to .fits files. This func is made for for images taken with the ASIStudio software using the ZWO camera.

    Args:
        png_file (str): path of .png (or .PNG) file.
        fits_file (str): path of .fits file
        metadata_file (None): if none, no header will be added. Note: if images are taken with  ASIStudio software using the SWO camera, the metadata file is of the form .PNG.txt. If providing self created files, it should be a .txt files wher values are denoted clearly in the form [name] = [value]. Each varirable should be on a new line.

    Raises:
        ValueError: _description_
    """    
    """

    Args:
        png_file (str): 
        metadata_file (str): path of metadata .txt (or .PNG.txt) file.
        fits_file (str): 
    """ 
    if not os.path.exists(fits_file):     
        # Read image
        image = Image.open(png_file)
        image_data = np.array(image)

        # Create FITS primary HDU
        hdu = fits.PrimaryHDU(image_data)

        # Read metadata and add to header
        if metadata_file != None: # if a path is given, add header
            if os.path.exists(metadata_file):
                metadata = read_metadata(metadata_file)
                for key, value in metadata.items():
                    hdu.header[key] = value
            else: raise ValueError("metadata_file does not exist.")
        #else no header

        # Write to FITS file
        hdu.writeto(fits_file, overwrite=True)
        print(f"FITS file '{fits_file}' created successfully.")
    else: print(f"skipping: FITS file '{fits_file}' already exists.")

def convert_png2fits_dir(fdir:str) -> None:
    """ converts dir of .pngs and .txt (metadata) to combined fits files. This func is made for for images taken with the ASIStudio software using the ZWO camera.

    Args:
        fdir (str): path to directory that holds dirs of .png files.
    """    
    subdirs = glob(os.path.join(fdir,'*'))
    for fdir in subdirs:
        imgdir = os.path.join(fdir,'*.PNG')
        metadir = os.path.join(fdir,'*.PNG.txt')
        fnames = glob(imgdir) 
        metafnames = glob(metadir) 
        print(len(fnames),len(metafnames))
        fnames.sort()
        metafnames.sort()
        for i in range(len(fnames)):
            png_file = fnames[i]
            metadata_file = metafnames[i]
            fits_file = png_file.replace('.PNG','.fits')
            png_to_fits(png_file, metadata_file, fits_file)
