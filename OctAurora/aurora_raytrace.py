#%%
import os
from pickletools import TAKEN_FROM_ARGUMENT1
from re import T
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

from hitmis_View.animate import clip_percentile

#%%
def get_tstamp(fn):return int(fn.strip('.fit').split('_')[-1])*0.001

# %% Oct Aurora Data HMS B - Back Window
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms2_OctAurora/*.fit'
fnames = glob(imgdir)
fnames = sorted(fnames,key=get_tstamp)
print(len(fnames))
fn = fnames[-1]
#%%
with fits.open(fn) as hdul:
    data = hdul[1].data
    header = hdul[1].header
#%%
cropped_data = data[100:-160,50:-68]
img_data = open_eclipse_data(cropped_data,crop_xpix=1,crop_ypix=1,imgsize = 1024,plot=True) 
#%%

predictor = HMS_ImagePredictor('bo',66.45,50, 90+2,1024)
# mapping = MapPixel2Wl(predicto
# s = mapping.straighten_img(wavelength=486.1, img=img_data, rotate_deg=1.25)
#%%
#[557.7, 630.0,427.8, 784.1,777.4,486.1,656.3,656.8,481,644,786.0,782.1,780.8,652.2,654.4,653.3, 774.4]
img = predictor.plot_spectral_lines('Detector',True,wls = [557.7, 630.0,427.8, 486.1], mosaic=True,measurement=True)
plt.axhline(predictor.g0,linewidth = .5, linestyle = '--',color ='orange')
plt.imshow(img_data,vmax = 1000,cmap = 'Greys')
plt.colorbar()
plt.show()


# %% Current HMS A img
######################################################################################################################################################

fdir = '../data/hms1_OctAurora/*.fits'
fnames = glob(fdir)
fnames.sort()
print(len(fnames))
#%%
predictor = HMS_ImagePredictor('a',67.4,50,mgammadeg=90)

#%%
fn = fnames[10]
with fits.open(fn) as hdul:
    data = hdul[1].data
    header = hdul[1].header
# %%
fig = predictor.plot_spectral_lines(ImageAt='MosaicWindow', Tape2Grating=True,wls=[557.7,486.1, 427.8, 557.7, 630, 656.3, 777.4],fprime = 449.70,measurement=True)
plt.axhline(predictor.g0,linewidth = .5, linestyle = '--',color ='orange')
# plt.axhline()
# plt.imshow(data,cmap = 'Greys',vmax = 570)
# plt.colorbar()
# %%
#%%
mapping = MapPixel2Wl(predictor)
#%%
simg,img,wlaxis = mapping.straighten_img(wavelength = 630.0, img = fn, rotate_deg=0.1) #rotation = 0.75
# %%
plt.imshow(simg,vmax = 800, aspect = 'auto')
plt.axvline(112, color = 'white', linewidth = 0.5)
# %%

