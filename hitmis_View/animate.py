#%% plots total counts in a panel over time. 
from __future__ import annotations 
from collections.abc import Iterable
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import tqdm
import  os
from glob import glob
from datetime import datetime
import subprocess
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %% Functions 
PATH = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(PATH,"../hms1_FoxHall_L1A/*")
wl = 4861.
if isinstance(wl, (int,float,str)):
    wl = int(wl)
    datadir += f'{wl}.nc'

files = glob(datadir)
# %%
def get_tindex(t:Iterable,start:datetime, end:datetime) -> Iterable: return np.where((t >= start) & (t<= end))[0]

# %%
#%%
fig,ax = plt.subplots(figsize=(5,5))
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
cax.set_label('Counts/Sec', rotation=270, labelpad=15)

ax.set_xlabel("Wavelegnth (nm)")


hh_mm = DateFormatter('%H:%M')


ax.set_title(f"{wl} nm Panel")

ims = []
for i in range(10):
    im = ax.imshow(ds.imgs[i].values,  aspect='auto')
    fig.colorbar(im, cax=cax)

    ax.set_xticks(np.arange(im.shape[1]))
    ax.set_yticks(np.arange(im.shape[0]))

    

    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)


writervideo = animation.FFMpegWriter(fps=5) 
ani.save(f'{wl}.mp4', writer=writervideo) 
plt.close() 
# plt.show()
# %%
fig, ax = plt.subplots(figsize = (5,5))
ax.imshow(ds.imgs[1].values, aspect = 'auto')
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
ax.set_xticks(np.arange(np.shape(ds.imgs[1].values)[-1])[::30], ds.imgs[1].wl.values[::30])
# %%

# Generate some example data
data = np.random.random((10, 10))


xticks = ds.imgs[1].wl.values




ax.set_yticklabels(np.arange(data.shape[0]))

# Format the x-tick labels to display only two decimal places
formatter = FuncFormatter(lambda x, pos: f'{xticks[int(x)]:.2f}')
ax.xaxis.set_major_formatter(formatter)

# Set the x-tick labels
ax.set_xticklabels([f'{val:.2f}' for val in xticks])

# Auto-adjust tick parameters
plt.xticks(rotation='auto')
plt.yticks(rotation='auto')

# Display the plot
plt.show()