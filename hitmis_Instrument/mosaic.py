#%%
import sys
sys.path.insert(0, '/home/charmi/Projects/hitmis_analysis/cppython')
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from glob import glob
from tqdm import tqdm
import Utils._files as _files
import astropy.io.fits as pf
import matplotlib.patches as patches
import pandas as pd
# %%


fn = _files.get_path('hms2_Mosaic/hms_c/', 'ccdi','.fits')

files = glob(fn)

# %%
with pf.open(files[0]) as hdul:
    img = hdul[1].data.astype(np.float64)
    print(hdul[1].header)
    exp = int(hdul[1].header['HIERARCH EXPOSURE_US'])*0.001
#%%
csv = pd.read_csv('edges.csv')
csv
# %%
ax,fig = plt.subplots()
plt.imshow(img, origin= 'lower')
plt.colorbar()
colors = ['orange','yellow', 'green', 'red', 'purple', 'white']
for i in range(csv.shape[0]):
    x = csv['xpix'][i]
    y = csv['ypix'][i]
    w = csv['widthpix'][i]
    h = csv['heightpix'][i]
    wl = csv['wlnm'][i]
    s = csv['xsizemm'][i]

    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=colors[i], facecolor='none')
    fig.add_patch(rect)
    plt.text(x+10,y+20,f'{wl} nm\n{s} mm\n{w} pix',color = colors[i],size = 8)

# %%
