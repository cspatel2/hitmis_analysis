#%%
import os
from pickletools import TAKEN_FROM_ARGUMENT1
import wave
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor, load_pickle_file
import astropy.io.fits as pf
from glob import glob
from astropy.io import fits
from hmspython.Utils._files import *
from hmspython.Utils._Utility import *

from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from skimage.transform import warp

#%%
def coord_trasformation(xy:np.array, wl_grid:np.ndarray, target_wls:np.array) -> np.ndarray:
    
    transformed_xy = np.empty(np.shape(xy))
    for i in range(len(xy)):
        row,col = int(xy[i,1]), int(xy[i,0])
        current_wl = wl_grid[row][col]
        target_col,_ =  find_nearest(target_wls, current_wl)
        transformed_xy[i,1] = row
        transformed_xy[i,0] = target_col
    return transformed_xy
    


# %% Solar Spectra / Eclipe data plot
# imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_EclipseRaw/EclipseDay/20240408/*fit'
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_SolarSpectra/20240507/*.fit'
fnames = glob(imgdir)
print(len(fnames))
fn = fnames[50]
#%%
# fn = fnames[2145] # eclipse

predictor = HMS_ImagePredictor('ae',67.32,50, 90-0.45)
mapping = MapPixel2Wl(predictor)
img_data = open_eclipse_data(fn)
#%%
s = mapping.straighten_img(wavelength=486.1, img=img_data, rotate_deg=1.25)
#%%
fig = predictor.plot_spectral_lines(ImageAt='detector', Tape2Grating=True,wls=[557.6,557.2,557.8,486.1, 427.8, 557.7, 630, 656.3, 777.4])
plt.imshow(img_data,cmap = 'Greys',vmin =0,vmax = 17000)
plt.colorbar()
plt.show()


# %% Current HMS A img
######################################################################################################################################################
fdir = '/home/charmi/Projects/hitmis_analysis/data/Images/hmsA_img/20240829/*.fits'
fnames = glob(fdir)
fnames.sort()
print(len(fnames))
predictor = HMS_ImagePredictor('a',67.39,50)
mapping = MapPixel2Wl(predictor)

fig = predictor.plot_spectral_lines(ImageAt='detector', Tape2Grating=True,wls=[557.6,557.2,557.8,486.1, 427.8, 557.7, 630, 656.3, 777.4])
plt.imshow(img_data,cmap = 'Greys')
plt.colorbar()
plt.show()
simg,img = mapping.straighten_img(mapping, wavelength = 656.3, img = fnames[1], rotate_deg=.75) #rotation = 0.75

#%%