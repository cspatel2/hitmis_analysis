
#%% find the pattern in timestamps jumble from aurora May 10, 2024.
from __future__ import annotations 
import types
import sys
from collections.abc import Iterable
from collections import Counter
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tqdm import tqdm
import  os
from glob import glob
from datetime import datetime
import subprocess
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import compress
#%%
def get_path(directory: str, prefix:str = None,suffix:str = None, wl:str =None )-> str:
    #build absolute path
    PATH = os.path.dirname(os.path.realpath(__file__))
    if isinstance(prefix, type(None)):
        dn = f'../{directory}/*'
    elif isinstance(prefix, str):
        dn = f'../{directory}/{prefix}*'
    else:
        sys.exit("Prefix must be a string or None")

    datadir = os.path.join(PATH,dn)
    
    if not isinstance(suffix, type(None)):
        datadir = dn + suffix
        if '.' not in suffix:
            datadir += '*'
    
    if isinstance(wl, (int,float,str)):
        wl = int(wl)
        datadir += f'{wl}.nc'
    return datadir

def get_tstamp(files:Iterable|str) -> Iterable:
    suffix = files[0].split('.')[-1]
    filetype = f'.{suffix}'

    if filetype == ".jpg": splitby = '/'
    elif filetype == '.fit': splitby = '_'
    else: sys.exit("File type must be .jpg or .fit")

    tstamps = [int(int(f.rstrip(filetype).split(splitby)[-1])*1e-3) for f in files]
    return tstamps

def datetime_from_tstamp(tstamp:int) -> datetime:
    if isinstance(tstamp, Iterable):
        datetime_ = [datetime_from_tstamp(t) for t in tstamp]
        return datetime_
    if isinstance(tstamp, str):
        tstamp = int(tstamp)
    return datetime.fromtimestamp(tstamp)

#%%

allfiles = glob(get_path('aurora_052024/all', suffix='.fit'))  
dayfiles = glob(get_path('aurora_052024/day/20240510', suffix='.fit'))
dayfiles += glob(get_path('aurora_052024/day/20240511', suffix='.fit'))
nightfiles = glob(get_path('aurora_052024/night/20240510', suffix='.fit'))
nightfiles += glob(get_path('aurora_052024/night/20240511', suffix='.fit'))

# %%

#timstamps
tstamp = get_tstamp(allfiles)
dtstamp = get_tstamp(dayfiles)
ntstamp = get_tstamp(nightfiles)

#datetime
dt = datetime_from_tstamp(tstamp)
ddt = datetime_from_tstamp(dtstamp)
ndt = datetime_from_tstamp(ntstamp)
#%%

#difference in consecutive timestamps
diff = np.diff(tstamp)
ddiff = np.diff(dtstamp)
ndiff = np.diff(ntstamp)



# %%
ts = [t-tstamp[0] for t in tstamp]
# %%
sorted_dt = [x for _, x in sorted(zip(diff, dt))]
# %%
didx = [tstamp.index(e) for e in dtstamp if e in tstamp]
nidx = [tstamp.index(e) for e in ntstamp if e in tstamp]

# %%
