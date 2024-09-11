#%%
import __future__
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from tqdm import tqdm
import  os
from glob import glob
from datetime import datetime
import subprocess

#%%
files = glob('/home/charmi/Projects/hitmis_analysis/hms1_FoxHall/20240604/*.fit')

def get_tstamp(fn:str)->int:
    return int((fn.rsplit('_')[-1].split('.')[0]))*1e-3


# %%
files = sorted(files,key=get_tstamp)

tstamp = []
count_s = []      
for file in tqdm(files[::10]):
    with pf.open(file) as hdul:
        img = hdul[1].data.astype(np.float64)
        exp = int(hdul[1].header['HIERARCH EXPOSURE_MS'])*0.001
        tstamp.append(int(hdul[1].header['HIERARCH TIMESTAMP']))
        count_s.append(img.sum()/exp) #image counts per sec
    

# %%
counts = np.array(count_s)
time =  np.array([datetime.fromtimestamp(t) for t in tstamp])
# %%
start = datetime(2024, 6, 4, 0, 0, 0)
end = datetime(2024, 6, 4, 4, 0, 0)
mask = list(np.where((time >=start) & (time <= end))[0] )
# %%
plt.plot(time[mask],counts[mask])
# plt.plot(time,counts)
plt.xlabel('time')
plt.ylabel("total Detector Counts / sec")
# %%
