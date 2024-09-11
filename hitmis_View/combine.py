
#%%
from __future__ import annotations 
from collections.abc import Iterable
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tqdm import tqdm
import  os
from glob import glob
from datetime import datetime
import subprocess
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

wls = [5577, 6300, 6563, 4861]

for wl in tqdm(wls):
    PATH = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(PATH,"../hms1_FoxHall_L1A/*")
    
    if isinstance(wl, (int,float,str)):
        wl = int(wl)
        datadir += f'{wl}.nc'

    files = glob(datadir)
    print( 'File Names Collected...')
    print('Loading ds1')

    ds1 = xr.load_dataset(files[-1])
    print('Loaded ds1')
    print('Loading ds2')
    ds2 = xr.load_dataset(files[0])
    print('Loaded ds2')

    try:
        print('combining datasets')
        # Concatenate the two datasets along the time dimension
        combined_ds = xr.concat([ds1, ds2], dim= 'tstamp')
    except Exception as e:
        print(f'Aborting combine_ds due to: {e}')
    
    print('datasets combined...')
    # Sort by the timestamp dimension
    print('SAve as new .nc files')
    # combined_ds = combined_ds.sortby('tstamp')

    words = files[-1].rstrip('.nc').split('_') #get the path to folder
    outfn = f'Summer_{words[-2]}_{words[-1]}.nc' # name of the file
    outpath = os.path.join(datadir.split('*')[0], outfn) # join to make final path

    # Save the combined dataset to a new file
    combined_ds.to_netcdf(outpath)
    print(f'Saved for combined file of {wl}')

