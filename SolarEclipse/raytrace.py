#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from hitmis_Instrument.img_predictor import HMS_ImagePredictor, load_pickle_file
import astropy.io.fits as pf
from glob import glob
from astropy.io import fits
from hmspython.Utils._files import *
from hitmis_Instrument.pix2wl_determination import MapPixel2Wl

# %%
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_EclipseRaw/EclipseDay/20240408/*fit'
# imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_SolarSpectra/20240507/*.fit'
fnames = glob(imgdir)
print(len(fnames))

#%%
# fn = fnames[2145] # eclipse
fn = fnames[100]
predictor = HMS_ImagePredictor('ae',67.32,50, 90-0.45)
#%%
mapping = MapPixel2Wl(predictor)
#%% Image at Detector
img_data = open_eclipse_data(fn,normalize=True)
fig = predictor.plot_spectral_lines(ImageAt='detector', Tape2Grating=True,wls=[557.6,557.2,557.8,486.1, 427.8, 557.7, 630, 656.3, 777.4])
plt.imshow(img_data,cmap = 'Greys')#,vmin =6000 ,vmax = 17000)
plt.colorbar()
plt.show()
# %%
wl = 486.1
simg,wlarr = mapping.straighten_img(wl,img=img_data,plot=True)
plt.figure()
plt.imshow(simg,aspect = 'auto')
idx = np.argmin(np.abs(wlarr-wl))
plt.axvline(x = idx)
wllabels = np.array([f'{x:.1f}' for x in wlarr])
l = plt.xticks(ticks = np.arange(0,len(wlarr),100),labels = wllabels[ np.arange(0,len(wlarr),100)])
# %%


