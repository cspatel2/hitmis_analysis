#%% plots total counts in a panel over time. 
from __future__ import annotations 
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

# %% Functions 
def get_path(directory: str, prefix:str = None,suffix:str = None, wl:str =None )-> str:
    #build absolute path
    PATH = os.path.dirname(os.path.realpath(__file__))
    if isinstance(prefix, type(None)):
        dn = f'../{directory}/*'
    elif isinstance(prefix, str):
        dn = f'../{directory}/{prefix}*'
    else:
        raise ValueError("Prefix must be a string or None")

    datadir = os.path.join(PATH,dn)
    
    if not isinstance(suffix, type(None)):
        print('checked nonetype')
        datadir = dn + suffix
        if '.' not in suffix:
            datadir += '*'
    
    if isinstance(wl, (int,float,str)):
        wl = int(wl)
        datadir += f'{wl}.nc'
    return datadir

def get_tindex(t:Iterable,start:datetime, end:datetime) -> Iterable: return np.where((t >= start) & (t<= end))[0]

# %%
def get_datadict(files: Iterable)-> dict:
    #initialize dictionary 
    Dict = {}

    #Get data from files
    for file in tqdm(files):
        wl = int(file.rstrip('.nc').split('_')[-1])
        ds = xr.open_dataset(file)
        ds_total_counts = ds.sum('height').sum('wl')  # total counts / sec in panel
        tc = ds_total_counts.imgs.values * ds_total_counts.exposure.values # total counts = count rate * exp
        
        if str(wl) in Dict.keys():
            Dict[str(wl)] += list(tc)
            Dict[f'{wl}_time'] += [datetime.fromtimestamp(int(i)) for i in ds_total_counts.tstamp.values]
        else:
            Dict[str(wl)] = list(tc)
            Dict[f'{wl}_time'] = [datetime.fromtimestamp(int(i)) for i in ds_total_counts.tstamp.values]
        del ds
    return Dict

def plot_totalcounts(datadict: dict, startT: datetime = None, endT: datetime = None, title:str = ''): 
    #Get list of wavelengths from data
    Dict = datadict.copy()
    wls = np.unique([i.rstrip("_time") for i in Dict.keys()])
    wls.sort()
    print(wls)

    #Slice data dictionary for the time range given
    if isinstance(startT , datetime) and isinstance(endT, datetime):
        Dict = {key: list(compress(values, ((startT <= t) and (t <= endT) for t in Dict[f'{wls[0]}_time']))) for key, values in Dict.items()} 

    # Get the date that the maximum number of data points is from
    def get_maxdate(dates:Iterable) -> datetime:
        # Get the date that the maximum number of data points is from
        dm = [dt.date() for dt in dates]
        date_counts = Counter(dm)
        return (max(date_counts, key=date_counts.get))
    
    date_label = get_maxdate(Dict[f'{wls[0]}_time'])
    print(date_label)
    
    #Create a figure
    fig, ax = plt.subplots()
    hh_mm = DateFormatter('%H:%M') #format the datetime objects to H:M
    ax.xaxis.set_major_formatter(hh_mm)
    
    ax.set_xlabel(date_label.strftime("%m-%d-%Y"))
    ax.set_ylabel('Total Counts')
    ax.set_title(title)

    #define colors for each wavelength. they correspond to the closest color of wl
    colors_dict = {'6300':'red', '5577':'green', '4861':'blue', '6563': 'darkred'}

    #plot
    if len(wls) == 1:
        ax.scatter(Dict[f'{wls[0]}_time'], Dict[wls[0]], label = f'{wls[0]} A', color = colors_dict[wls[0]], s = 1.5)
    else:
        for wl in wls:
            ax.scatter(Dict[f'{wl}_time'], Dict[wl], label = f'{wl} A', color = colors_dict[wl], s = 1.5)
    
    ax.legend(loc = 'best')
    plt.show()

# %%
    
fpath = get_path('hms1_FoxHall_L1A',prefix = 'Summer')
# fpath = get_path('aurora_052024', 'AuroraNight' )
files = glob(fpath)        
files.sort()
#%%
# Example usage:
data = get_datadict(files)
#%%
# plot_totalcounts(data, title = 'Nighttime May Aurora')
plot_totalcounts(data, startT= datetime(2024,6,4,21,20,0), endT= datetime(2024,6,4,23,59,0), title='Night Fox Hall')

#





# %%
