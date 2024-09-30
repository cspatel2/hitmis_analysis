#%% find the spectral resolution at different slit widths
import matplotlib.pyplot as plt
import numpy as np
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from hmspython.Utils._Utility import correct_unit_of_angle
# %%
predictor = HMS_ImagePredictor(hmsVersion='a',alpha=67.39)
predictor.plot_spectral_lines('Detector',True)
# %%
mapping = MapPixel2Wl(predictor)
# %%
alpha = np.array(mapping.alphagrid)
beta = np.array(mapping.betagrid)
gamma = np.array(mapping.gammagrid)
panel = np.array(mapping.panelgrid)

wls = np.array(mapping.lambdagrid)

#%%
resolution = mapping.get_resolution_grid(slitwidth=50)


# %%
value = mapping.extract_wlpanel(5577,resolution)
# %%
np.mean(value)
# %%
