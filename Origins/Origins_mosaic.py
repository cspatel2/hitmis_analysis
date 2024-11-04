# %%
import os
import sys
import wave
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor, load_pickle_file
import astropy.io.fits as pf
from astropy.io import fits
from hmspython.Utils._files import *
from hmspython.Utils._Utility import *
from tqdm import tqdm

# %%
predictor = HMS_ImagePredictor(config'/home/charmi/Projects/Hitmis/hmsdesigner/configs/hmsa_origin.toml')
# img = predictor.plot_spectral_lines('MosaicWindow', True, wls=[
#                                     557.7, 630.0, 427.8, 784.1, 777.4, 486.1, 485, 656.3, 656.8, 644, 786.0, 782.1, 780.8, 652.2, 654.4, 653.3], mosaic=True, measurement=True)
# plt.axhline(predictor.g0, linewidth=.5, linestyle='--', color='orange')
#%%
img = predictor.plot_spectral_lines('MosaicWindow', True, wls=[486.1], mosaic=True, measurement=True)
plt.axhline(predictor.g0, linewidth=.5, linestyle='--', color='orange')
#%%
# xlim = plt.xlim()
# locs = [min(xlim), 517, 524.5, 532.5]
# for loc in locs[1:]:
#     # plt.axvline(loc, 0.5, 1)
#     pass
# print('Top:', np.diff(locs), max(xlim) - max(locs))

# locs = [min(xlim), 515]
# for loc in locs[1:]:
#     # plt.axvline(loc, 0, 0.5)
#     pass
# print('Bottom:', np.diff(locs), max(xlim) - max(locs))
# plt.show()
# sys.exit(0)
# plt.axvline(465, 0 ,0.5)
# plt.axvline(449,0.5,1)
# plt.axvline(475,0.5,1)
# plt.axvline(483,0.5,1)


# fig = predictor.plot_spectral_lines(ImageAt='Detector', Tape2Grating=True,wls=[557.7,486.1, 427.8, 557.7, 630, 656.3, 777.4],fprime = 442.9,measurement=True)
# plt.axhline(predictor.g0,linewidth = .5, linestyle = '--',color ='orange')

# %%
## Test the new relslitposition with current mosaic

predictor = HMS_ImagePredictor('/home/charmi/Projects/hitmis_analysis/hmspython/hmspython/Utils/hmsa_aurora_slittest.json', 67.455, 50, mgammadeg=90-.05, pix=1024)
# img = predictor.plot_spectral_lines('MosaicWindow', True, wls=[
#                                     557.7, 630.0, 427.8, 784.1, 777.4, 486.1, 485, 656.3, 656.8, 644, 786.0, 782.1, 780.8, 652.2, 654.4, 653.3], mosaic=True, measurement=True)
# plt.axhline(predictor.g0, linewidth=.5, linestyle='--', color='orange')


# %%
fnames = glob('2024-10-22_11_40_39Z/Light_*.fit')
impath = fnames[1]
with pf.open(impath) as hdul:
    data = hdul[0].data.astype(np.float64)
    
#resize img
IMGSIZE = 1024
scale_factor = (IMGSIZE / data.shape[0], IMGSIZE / data.shape[1])
data = transform.rescale(data, scale_factor, order=3)
#rotate img
data = transform.rotate(data,-0.1)


# %%
img = predictor.plot_spectral_lines('Detector', True, wls=[557.7, 630.0, 427.8, 777.4, 486.1, 656.3, 486.1], mosaic=True, measurement=True)
vmin = np.percentile(data,1)
vmax = np.percentile(data,99)
plt.imshow(data, cmap = 'pink', vmin = vmin, vmax = vmax)
plt.colorbar()

# %%
config = '/home/charmi/Projects/hitmis_analysis/HitmisL1Processor/configs/hmsa_aurora.json'
predictor = HMS_ImagePredictor(config, 67.43, 50, mgammadeg=90-.05, pix=1024)
img = predictor.plot_spectral_lines('MosaicWindow', True, wls=[
                                    557.7, 630.0, 427.8, 784.1, 777.4, 486.1, 485, 656.3, 656.8, 644, 786.0, 782.1, 780.8, 652.2, 654.4, 653.3], mosaic=True, measurement=True)
# %%
config = '/home/charmi/Projects/hitmis_analysis/HitmisL1Processor/configs/hmsa_aurora.json'
predictor = HMS_ImagePredictor(config, 67.43, 50, mgammadeg=90-.05, pix=1024)
img = predictor.plot_spectral_lines('MosaicWindow', True, wls=[486.1], mosaic=True, measurement=True)
# %%
