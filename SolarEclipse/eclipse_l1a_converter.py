#%% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

#%%
import matplotlib.pyplot as plt
import numpy as np 
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from hmspython.Utils import _files
from hmspython.Utils import _Utility

# %%
