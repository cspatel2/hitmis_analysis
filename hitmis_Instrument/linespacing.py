#%%
import sys
import os
from Utils._files import open_fits,get_path
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from glob import glob
from tqdm import tqdm
import astropy.io.fits as fits
import matplotlib.patches as patches
import pandas as pd
from scipy.signal import find_peaks
# %%
# fn = get_path('hms2_align/hydrogen/', 'ccdi','.fits')
fn = get_path('hms2_align/night/', 'ccdi','.fits')

files = glob(fn)
files.sort()
# %%
import astropy.io.fits as fits
import xarray as xr
import numpy as np
#%%
# Open the FITS file
data,header = open_fits(files[-3])
# %%
plt.imshow(data,vmin = 0,vmax = 1500)
plt.colorbar()
gain = header['gain']
exposure = header['HIERARCH EXPOSURE_US']*1e-6
plt.title(f'Gain = {gain}, Exposure = {exposure} s')

#%%
middle = data[1000:1020][:]
plt.imshow(middle,aspect = 'auto')
y = np.sum(middle, axis = 0)
x = np.arange(len(y))
# %%
peaks,heights = find_peaks(y,300000)

plt.plot(x,y)
for i in peaks:
    plt.scatter(x[i],y[i], marker = '.', color = 'red')

# %%
sizemm = 52.05 #mm
sizepix = 3004 # pixels
resolution = sizemm/sizepix #mm/pixel
focal_length = 400 #mm
dbeta_mm = resolution * (x[peaks[1]] - x[peaks[0]])
dbeta_deg = np.rad2deg(dbeta_mm / focal_length)
# %%
