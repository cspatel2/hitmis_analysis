#%% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

#%%
import matplotlib.pyplot as plt
import numpy as np 
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from hmspython.Utils import _files
from hmspython.Utils import _Utility
from glob import glob
import astropy.io.fits as fits

# %%
#for each file 

# 1. open .fits file
    #collect img, exposure, timestamp,gain
# 2. straighten img
# 3. store in a ds that can be saved as 
#%% 
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_SolarSpectra/20240507/*.fit'
fnames = glob(imgdir)
print(len(fnames))
fn = fnames[50]
#%%
# #%%

with fits.open(fn) as hdul:
    header = hdul[1].header
    tstamp = int(header['HIERARCH TIMESTAMP']) #ms
    time =  _files.time_from_tstamp(tstamp) #datetime object
    exposure = int(header['HIERARCH EXPOSURE_MS']) #ms
    gain = header['GCOUNT']
    camtemp = float(header['CCDTEMP']) #degrees Celcius
    data = hdul[1].data
    data = _files.open_eclipse_data(data) #crop and resize img to 3008x3008
    data =data/(exposure*1e-3) # [ADU/s]
    
    
    
# %%
