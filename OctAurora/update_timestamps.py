# %% Imports
from __future__ import annotations
from glob import glob
from typing import Optional
import astropy.io.fits as pf
import numpy as np
from tqdm import tqdm
import xarray as xr
import os
import re
# %% Setup
def get_date(name: str) -> Optional[str]:
    if os.path.splitext(name)[-1] == '.nc':
        date = re.findall(r'(\d{8})', name)
        if len(date) != 1:
            return None
        return date[0]
    return None

def getctime(name: str) -> int:
    with pf.open(name) as hdu:
        return int(hdu[1].header['TIMESTAMP'])
# %% Get the dates

fitsdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_Oct10Aurora'
ncdir = '/home/charmi/Projects/hitmis_analysis/OctAurora'

ncfiles = glob(f'{ncdir}/*.nc')
dates = list(set(list(map(get_date, ncfiles))))
dates.sort()
# %% Get the FITS files
comp = dict(zlib=True, complevel=5)
for date in tqdm(dates, position = 0):
    fitsfiles = glob(f'{fitsdir}/{date}/*.fit*')
    if len(fitsfiles) == 0:
        print('No files in', f'{fitsdir}/{date}')
        break
    fitsfiles.sort()
    tstamps = list(map(getctime, fitsfiles))
    ncs = glob(f'{ncdir}/Aurora_*{date}*.nc')
    for ncfile in tqdm(ncs, position = 1):
        nds = None
        with xr.load_dataset(ncfile) as ds:
            nds = ds.copy(deep=True)
        nds.coords['tstamp'] = np.asarray(tstamps)
        encoding = { var: comp for var in list(ds.data_vars) + list(ds.coords) }
        outfile = ncfile.replace('Aurora_', 'AuroraFixed_')
        nds.to_netcdf(outfile, encoding=encoding)



# %%
