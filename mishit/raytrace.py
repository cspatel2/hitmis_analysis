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
predictor = HMS_ImagePredictor('mishit.json', 55, 150, mgammadeg=90, pix=1024)
# img = predictor.plot_spectral_lines('MosaicWindow', True, wls=[
#                                     557.7, 630.0, 427.8, 784.1, 777.4, 486.1, 485, 656.3, 656.8, 644, 786.0, 782.1, 780.8, 652.2, 654.4, 653.3], mosaic=True, measurement=True)
# plt.axhline(predictor.g0, linewidth=.5, linestyle='--', color='orange')
#%%
img = predictor.plot_spectral_lines('Detector', True, wls=[557.7, 630, 427.8, 777.4, 784.1, 782.1, 780.8], mosaic=True, measurement=True)
plt.axhline(predictor.g0, linewidth=.5, linestyle='--', color='orange')
# %%
