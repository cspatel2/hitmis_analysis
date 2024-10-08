#%% 
from __future__ import annotations 
from collections.abc import Iterable
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import  os
from glob import glob
from datetime import datetime
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import difflib

# %% Functions 


# %%
def clip_percentile(image, lower_percentile=1, upper_percentile=99):
    """Clips the image to the specified percentiles."""
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    return np.clip(image, lower_bound, upper_bound)
#%%
# %%
def animate(files: list, start_time:np.datetime64 = None,end_time:np.datetime64 = None, title_prefix: str = '',save_filename: str = 'anim.mp4',bitmode = 16 ):
    
    if start_time == None: 
        with fits.open(files[0]) as hdul:
            header = hdul[1].header
            tstamp = header['TIMESTAMP']*0.001
            start_time = datetime.fromtimestamp(tstamp)
            
    if end_time == None: 
        with fits.open(files[-1]) as hdul:
            header = hdul[1].header
            tstamp = header['TIMESTAMP']*0.001
            end_time = datetime.fromtimestamp(tstamp)
    
    with fits.open(fnames[0]) as hdul:
        header = hdul[1].header
    expstr = difflib.get_close_matches('EXPOSURE', list(header))[0]
    if '_US' in expstr: to_sec = 1e-6
    elif '_MS' in expstr: to_sec = 1e-3
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    frames = []
    times = []
    if bitmode < 16: bitscale = 2^bitmode
    else: bitscale = 1
    for i,fn in enumerate(tqdm(files)):
        with fits.open(fn) as hdul:
            data = hdul[1].data
            data = clip_percentile(data)
            header = hdul[1].header
            tstamp = header['TIMESTAMP']*0.001
            exposure = header[expstr]*to_sec
        time = datetime.fromtimestamp(tstamp)
        if time >= start_time and time <= end_time: #post sunset
            times.append(time.strftime("%Y-%m-%d %H:%M:%S"))
            frame = (data)/exposure
            frame = np.asarray(frame)*bitscale
            frames.append(frame)
        
    f0 = frames[0]
    t0 = times[0]
    im = ax.imshow(f0)
    cbar = fig.colorbar(im, cax=cax, label='Counts/s')
    tx = ax.set_title('{title} : {time}'.format(title = title_prefix,time = t0))

    def animate_update(i) -> list:
        arr = frames[i]
        im.set_data(arr)
        im.set_clim(vmin=arr.min(), vmax=arr.max())
        timestr = times[i]
        tx.set_text('{title} : {time}'.format(title = title_prefix,time = timestr))
        cbar.update_normal(im)
        return [im, tx, cbar]
    
    anim = animation.FuncAnimation(fig, animate_update, frames=len(frames), interval=1000, repeat_delay=1000, blit=False,)
    anim.save(save_filename,fps = 5,extra_args=['-vcodec', 'libx264']) 

    print('Done!')


# %%
################ HMS B OCT AURORA #################################
# datadir = "../data/hms2_OctAurora/*.fit"
# def get_tstamp(fn):return int(fn.strip('.fit').split('_')[-1])*0.001
# fnames = glob(datadir)
# fnames = sorted(fnames,key=get_tstamp)
# print(len(fnames))

# name = 'OctAurora_hms2_20241006_postsunset.mp4'
# start = datetime(2024,10,6,18,20,0)
# animate(fnames,start,None,'HiT&MIS B',name) 
#%%
# ############### HMS A OCT AURORA #################################
datadir = "../data/hms1_OctAurora/*.fits"
fnames = glob(datadir)
fnames.sort()
print(len(fnames))

name = 'OctAurora_hms1_20241006_postsunset.mp4'
start = datetime(2024,10,6,18,20,0)
end = datetime(2024,10,6,23,54,5)

animate(fnames,start,None,'HiT&MIS A',name, bitmode=8)



    
    

# %%
