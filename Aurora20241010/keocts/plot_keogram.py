#%%
from cProfile import label
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import xarray as xr
from datetime import datetime
import matplotlib.dates as mdates
# %%
ds = xr.load_dataset('hmsa_20241010_6300.nc')
# %%
ds
# %%
ds.tstamp.fromtimestamp()
# %%
x = ds.tstamp.values
x = [datetime.fromtimestamp(t) for t in x]
y = ds.height.values
z = ds['6300'].values.transpose()

# %%
fig, ax = plt.subplots(figsize=(9, 4), dpi=300, tight_layout=True)
# plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['font.monospace'] = 'Andale Mono'
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)
locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
X,Y = np.meshgrid(x,y)
img = ax.pcolor(X,Y,z)
ax.set_ylabel('Detect Pixel Y')
fig.colorbar(img,label = 'ADU/s')
ax.set_title("HiT&MIS A\n ROI: 630.0 nm")

plt.savefig('hms1_Aurora_keogram_20241010_6300.png',dpi = 300)





# %%


# %%
