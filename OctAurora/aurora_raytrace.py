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
# import skimage.transform 
from skimage import transform

#%%
def get_tstamp(fn):return int(fn.strip('.fit').split('_')[-1])*0.001

# %% Oct Aurora Data HMS B - Back Window
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms2_OctAurora/*.fit'
fnames = glob(imgdir)
fnames = sorted(fnames,key=get_tstamp)
print(len(fnames))
fn = fnames[2075]
# fn = fnames[2]
#%%

#load
with fits.open(fn) as hdul:
    data = hdul[1].data
    header = hdul[1].header
#crop
data = data[100:-160,50:-68]
#resize to 1024
zoom_factor = (1024 / data.shape[0], 1024/ data.shape[1])
data = zoom(data, zoom_factor, order=3)

#rotate
data = transform.rotate(data,-1,cval = np.nan, preserve_range=True)

#mirror
data = np.flip(data, axis=(0, 1))


#%%
predictor = HMS_ImagePredictor('b',65.85,50, 90-1,1024)
# mapping = MapPixel2Wl(predicto
# s = mapping.straighten_img(wavelength=486.1, img=img_data, rotate_deg=1.25)
#%%
#[557.7, 630.0,427.8, 784.1,777.4,486.1,656.3,656.8,481,644,786.0,782.1,780.8,652.2,654.4,653.3, 774.4]
img = predictor.plot_spectral_lines('Detector',True,wls = [557.7, 630.0,656.3,486.1,777.4,427.8], mosaic=True,measurement=True,fprime = 350)

plt.axhline(predictor.g0,linewidth = .5, linestyle = '--',color ='orange')
plt.imshow(data,cmap = 'Greys',vmax = 2000)
plt.colorbar()
plt.show()

#%%
# %% Current HMS A img
######################################################################################################################################################


# get a list of files
fdir = '../data/hms1_OctAurora/*.fit*'
fnames = glob(fdir)
fnames.sort()
print(len(fnames))
#%%
#initialize predictor
predictor = HMS_ImagePredictor('a',67.43,50,mgammadeg=90-.05,pix = 1024)

#open file
fn = fnames[392]
# fn = fnames[20]
print(fn)
with fits.open(fn) as hdul:
    data = hdul[1].data
    header = hdul[1].header
    tstamp = header['TIMESTAMP']*0.001
    timestr = datetime.fromtimestamp(tstamp)
    print(timestr)

#resize to 1024
zoom_factor = (1024 / data.shape[0], 1024/ data.shape[1])
resized_img_data = zoom(data, zoom_factor, order=3)



#overlay img and model
data = resized_img_data
vmin = np.percentile(data,1)
vmax = np.percentile(data,99)
# fig = predictor.plot_spectral_lines(ImageAt='Detector', Tape2Grating=True,wls=[557.7,486.1, 427.8, 557.7, 630, 656.3, 777.4],fprime = 442.9,measurement=True)
img = predictor.plot_spectral_lines('Detector',True,wls = [557.7, 630.0,427.8, 784.1,777.4,486.1,656.3,656.8,481,644,786.0,782.1,780.8,652.2,654.4,653.3, 774.4], mosaic=True,measurement=True)
plt.axhline(predictor.g0,linewidth = .5, linestyle = '--',color ='orange')
plt.imshow(data,cmap = 'rainbow',vmin = vmin, vmax = 50)
plt.colorbar()
#%%
#Inititalize mapping
mapping = MapPixel2Wl(predictor)

#Straighten img
wl = 630.0
simg,img,wlaxis = mapping.straighten_img(wavelength = wl, img = data, rotate_deg = .1) #rotation = 0.75

#plot a bigger version of straightened img
plt.imshow(simg,vmax = 100, aspect = 'auto')
plt.axvline(find_nearest(wlaxis,wl)[0], color = 'white', linewidth = 0.5)
plt.colorbar()
plt.title(f'{wl} nm')
# %%


#%%
#initialize predictor
predictor = HMS_ImagePredictor('a',67.43,50,mgammadeg=90-.05,pix = 1024)


fig = predictor.plot_spectral_lines(ImageAt='Detector', Tape2Grating=True,wls=[557.7,486.1, 427.8, 557.7, 630, 656.3, 777.4],fprime = 442.9,measurement=True)
plt.axhline(predictor.g0,linewidth = .5, linestyle = '--',color ='orange')

#%%

img = predictor.plot_spectral_lines('Detector',True,wls = [557.7, 630.0,427.8, 784.1,777.4,486.1,656.3,656.8,481,644,786.0,782.1,780.8,652.2,654.4,653.3, 774.4], mosaic=True,measurement=True)

# %%
import xarray as ar
# %%
ds = ar.open_dataset('/home/charmi/Projects/hitmis_analysis/OctAurora/Aurora_20241012_6300.nc')
# %%
ds
# %%
ds.img[1000].plot()

# %%
