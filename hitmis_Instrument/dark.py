#%%
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
from hmspython.Utils._files import *
#%%
fdir = '/home/charmi/Projects/hitmis_analysis/data/hms_apollocam_dark/CapObj'



# %%
fits_file
# %%
with fits.open(fits_file) as hdul:
    data = hdul[0].data

# %%
convert_png2fits(fdir)

# %%
