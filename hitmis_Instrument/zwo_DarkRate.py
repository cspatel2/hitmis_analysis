#%%
from cProfile import label
from typing import LiteralString
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
from hmspython.Utils._files import *
import pickle
import lzma
import os
import astropy.io.fits as pf
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
#%%
def open_fitsfile(file:str) -> np.array:
    with pf.open(file) as hdul: 
        data = hdul[0].data.astype(np.float64)
    return data

def get_exposure_from_fits(file:str) -> float:
    with pf.open(file) as hdul: 
        exposure = hdul[0].header['EXPOSURE']
        exp,unit = separate_numbers_letters(exposure)
        if unit == 'ms': exp = int(exp)*1e-3 #ms -> s
        elif unit == 'us': exp = int(exp)*1e-6 #us -> s
        else: exp = int(exp) # already in s
    return exp

def separate_numbers_letters(string:str) -> tuple[LiteralString, LiteralString]:
    numbers = []
    letters = []
    for char in string:
        if char.isdigit():
            numbers.append(char)
        elif char.isalpha():
            letters.append(char)
    return ''.join(numbers), ''.join(letters)
#%%
fdir = '/home/charmi/Projects/hitmis_analysis/data/hms_apollocam_dark/CapObj'
convert_png2fits_dir(fdir) #conver pngs to fits files 

# %%
subdirs = glob(os.path.join(fdir,"*"))
subdirs.sort()
len(subdirs)

# %% Get mean and std for each exposure collection
meanpix = []
stdpix = []
exposures = []

for sub in subdirs:
    files = glob(os.path.join(sub,'*fits'))
    pixval = list(map(open_fitsfile, files))
    exps = list(map(get_exposure_from_fits,files))
    meanpix.append(np.mean(pixval, axis=0))
    stdpix.append(np.std(pixval, axis=0))
    exposures.append(np.mean(exps))


# %%
exposure = np.array(exposures)
meanpix = np.array(meanpix)
stdpix = np.array(stdpix)

dark = np.zeros(meanpix.shape[1:])
dark_std = np.zeros(meanpix.shape[1:])
bias = np.zeros(meanpix.shape[1:])
bias_std = np.zeros(meanpix.shape[1:])

for i in tqdm(range(meanpix.shape[1])):
    for j in range(meanpix.shape[2]):
        p, cov = curve_fit(lambda x, m, b: m*x+b, exposure,
                        meanpix[:, i, j], sigma=stdpix[:, i, j], absolute_sigma=True, p0=[0, 100])
        dark[i, j] = p[0]
        dark_std[i, j] = np.sqrt(cov[0, 0])
        bias[i, j] = p[1]
        bias_std[i, j] = np.sqrt(cov[1, 1])


# %%
data = dark
data_std = dark_std
vmin = np.mean(data) - 3 * np.mean(data_std)
vmax = np.mean(data) + 3 * np.mean(data_std)
plt.imshow(data, cmap='gist_ncar',vmin = vmin,vmax = vmax)
plt.colorbar(label='Dark Rate [Counts/sec]')
plt.xlabel('Pixel Postion X')
plt.ylabel('Pixel Postion Y')
plt.title('ZWO ASI432mm')
plt.show()
#%%
data = bias
data_std = bias_std[np.where(np.isfinite(bias_std))]
vmin = np.mean(data) - 3 * np.mean(data_std)
vmax = np.mean(data) + 3 * np.mean(data_std)
plt.imshow(data, cmap='gist_ncar',vmin = vmin,vmax = vmax)
plt.colorbar(label='Bias [Counts]')
plt.xlabel('Pixel Postion X')
plt.ylabel('Pixel Postion Y')
plt.title('ZWO ASI432mm')
plt.show()
# %%

# with lzma.open('../hitmis_l1_converter/pixis_dark_bias.xz', 'rb') as f:
#     data = pickle.load(f)
#     plt.imshow((data['dark']), cmap='gray')
#     plt.colorbar()
#     plt.show()

# %%
hist = plt.hist(dark.flatten(),100,(-0.3,0.3),True)
# %%

# %%
