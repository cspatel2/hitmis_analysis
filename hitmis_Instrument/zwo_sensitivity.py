# %%
from hmspython.Utils._files import *
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage import exposure
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor


# %%
fdir = '/media/windowsshare/hmsA/20240824/*.fits'
# %%
fnames = glob(fdir)
fnames.sort()
# %%
idx = 70
#%%
data,header = open_fits(fnames[idx])
ts = fnames[idx].split('ccdi_')[-1].strip('.fits')

# %%

data = exposure.equalize_hist(data)
plt.imshow(data,cmap ='gray')
plt.colorbar()
plt.title(format_time(ts))
# %%

# %%

predictor = HMS_ImagePredictor('a',67.39)
#
# %%
predictor.overlay_on_Image(fnames[idx])
# %%
