#%%
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
fdir = '../hmspython/Diffraction/hmsParams/'
#%% 
#--slit 1 (Dark Blue/purple)----------------------------------------------
fname = os.path.join(fdir,'FGB25.xlsx')
wl = 427.8

# #--slit 2 (Yellow)---------------------------------------------
# fname = os.path.join(fdir,'FGL495.xlsx')
# wl = [557.7,630.0,777.8]

#--slit 3 (Red)----------------------------------------------
# fname = os.path.join(fdir,'FGL590.xlsx')
# wl = [656.3]

# #--slit 4 (Blue)----------------------------------------------
# fname = os.path.join(fdir,'FGB7.xlsx')
# wl = [486.1,488.1]


dfs = pd.read_excel(fname)
x = list(dfs['Wavelength (nm)'])
y = list(dfs['% Transmission'])
f = interp1d(x,y,'linear')
f(wl)
# %%
# %%
