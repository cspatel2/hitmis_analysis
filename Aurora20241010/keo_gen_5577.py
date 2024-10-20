# %% Imports

# Fit the downsampled data to Gaussians (night time, 6 pm - 5 am) to extract feature brightness 5577A

import datetime as dt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from BaselineRemoval import BaselineRemoval
from scipy.optimize import curve_fit
from pysolar import solar
import pytz
import pandas
from matplotlib.pyplot import cm

from skmpython.GenericFit import GenericFitFunc, GenericFitManager
from skmpython import vac2air

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import uncertainties as un
from tqdm import tqdm
import matplotlib
rc('font',**{'family':'serif','serif':['Times New Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
# %%
background_fcn = lambda x, x0, a0, a1, a2: a0 + a1 * (x-x0) + a2*(x-x0)**2
fitfunc = GenericFitFunc(background_fcn=background_fcn, num_background_params=4)
feature_fcn = lambda x, c, a, w: a*np.exp(-((x-c)/w)**2)
fitfunc.register_feature_fcn(fcn=feature_fcn, num_params=3) # register feature 0, emission
fitfunc.finalize()
# %%
inpath = 'keosrc'
outdir = 'keodata'
try:
    os.mkdir(outdir)
except FileExistsError:
    pass
# %%
def get_wl(name):
    name = name.split('.nc')[0]
    wl = name.split('_')[-1]
    if len(wl) != 4:
        raise ValueError('%s not valid wavelength: has length %d'%(wl, len(wl)))
    return int(wl)

def get_date(name):
    name = name.split('.nc')[0]
    date = name.split('_')[-2]
    if len(date) != 8:
        raise ValueError('%s not valid date: has length %d'%(date, len(date)))
    return date
# %%
files = glob.glob('%s/*.nc'%(inpath))
dates = list(map(get_date, files)) # [get_date(f) for f in files]
wls = list(map(get_wl, files))
dates = list(set(dates))
wls = list(set(wls))
dates.sort()
wls.sort()
print(dates)
# %%
for date in dates:
    files = glob.glob('%s/hitmis_downsamp_%s_5577.nc'%(inpath, date))
    start = dt.datetime.strptime(date + ' 18:00:00', '%Y%m%d %H:%M:%S')
    end = dt.datetime.strptime(date + ' 05:00:00', '%Y%m%d %H:%M:%S') + dt.timedelta(days=1)
    outfile = '%s/hitmis_counts_%s_5577.nc'%(outdir, date)
    logfname = '%s/hitmis_counts_%s_5577.log'%(outdir, date)
    outimgdir = '%s/%s'%(outdir, date)
    try:
        os.mkdir(outimgdir)
    except FileExistsError:
        pass
    outimgnamebase = '%s/%s/hitmis_counts_%s_5577'%(outdir, date, date)
    logfile = open(logfname, 'w')
    if os.path.exists(outfile):
        print('File exists for %s - %s. Continuing.'%(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')))
        continue
    ds = xr.load_dataset(files[0])
    ds = ds.loc[dict(tstamp=slice(start, end), wl=slice(557.2, 557.85))]
    tstamp = ds.tstamp.values
    height = ds.height.values
    counts = np.zeros(ds.imgs.shape[:-1], dtype=float)
    dcounts = counts.copy()
    if (len(ds.tstamp)) == 0:
        print('No data between %s and %s... Continuing.'%(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')))
        continue
    trange = tqdm(range(len(ds.tstamp)))
    for tidx in trange:
        ctime = pandas.to_datetime(ds.tstamp[tidx].values).to_pydatetime()
        img = ds.imgs[tidx]
        std = ds.stds[tidx]
        x: np.ndarray = img.wl.values
        for idx in range(img.shape[0]):
            if idx % 5 == 0: trange.set_description(desc='Working on %s (%02d/%d)'%(ctime.strftime('%Y-%m-%d %H:%M'), idx, len(ds.height)))
            y: np.ndarray = img.values[idx, :]
            dy: np.ndarray = std.values[idx, :]
            p0 = [x.mean(), y.mean(), (y[-1] - y[0]) / (x[-1] - x[0]), 0] # x0, mean, slope, 0
            p_low = [-np.inf, 0, -np.inf, -np.inf]
            p_high =  [np.inf, np.inf, np.inf, np.inf]
            p0 += [557.7, y.max() - y.min(), 0.1] # 557.7 nm
            p_low += [557.6, 0, 0.02] # 557.7 nm
            p_high += [557.8, y.max(), 0.16] # 557.7 nm
            gfit = GenericFitManager(x, y, p0=p0, baseclass=fitfunc, figure_title='%d'%(idx), plot=False)
            try:
                popt, pcov = gfit.run(ioff=True, close_after=True, plot_every=500, bounds=(p_low, p_high), sigma=np.abs(dy), absolute_sigma=True)
                amp = un.ufloat(popt[5], np.sqrt(pcov[5, 5]))
                wid = un.ufloat(popt[6], np.sqrt(pcov[6, 6]))
                ct = np.sqrt(np.pi)*amp*wid
                counts[tidx, idx] = un.nominal_value(ct)
                dcounts[tidx, idx] = un.std_dev(ct)
            except Exception as e:
                counts[tidx, idx] = np.nan
                dcounts[tidx, idx] = np.nan
                logfile.write(f'{tidx}, {idx}\n')
                logfile.flush()
                fig, ax = plt.subplots(1, 1)
                ds['imgs'][tidx].plot(ax=ax)
                ds['imgs'][tidx, idx].plot(ax=ax.twinx(), color='k')
                plt.savefig(f'{outimgnamebase}_{tidx}_{idx}.png')
                plt.close()
                gfit = GenericFitManager(x, y, p0=p0, baseclass=fitfunc, figure_title='%d'%(idx))
                try:
                    popt, pcov = gfit.run(ioff=True, close_after=True, plot_every=1000, bounds=(p_low, p_high), sigma=np.abs(dy), absolute_sigma=True)
                    plt.close('all')
                except Exception as e:
                    print('\n\n', e, '\n')
                    pass
            # print('Index: %d | Iteration: %d | Center: %.3f nm, Amplitude: %.2e counts/nm, Width: %.2e nm | Strength: %s'%(idx, gfit.iterations, popt[4], popt[5], popt[6], str(ct)))
    logfile.close()
    ds2 = xr.Dataset(
        data_vars={'5577': (('tstamp', 'height'), counts, {'units': 's^{-1}'}),
                    '5577_std': (('tstamp', 'height'), dcounts, {'units': 's^{-1}'})},
        coords={'tstamp': tstamp, 'height': height}
    )
    ds2['height'].attrs.update({'units': 'rad'})
    ds2.to_netcdf(outfile)

# %%
sys.exit(0)
# %%
sol = xr.load_dataset('sao_2010_solref.nc')
# %%
sol
# %%
sol = sol.assign_coords(coords={'wavelength': vac2air(sol.wavelength.values)})
# %%
s63 = sol.spectral_irradiance.loc[dict(wavelength=slice(629.8, 630.9))]
# %%
for idx in range(img.shape[0]):
    x = img.wl
    y = img.values[idx, :]
    plt.plot(x, y)
plt.show()

for idx in range(img.shape[0]):
    x = img.wl
    y = std.values[idx, :]
    plt.plot(x, y)
plt.show()
# %%
from findpeaks import findpeaks

s63 = sol.spectral_irradiance.loc[dict(wavelength=slice(629.8, 630.9))]
s63_wl = s63.wavelength.values
s63_si = s63.values
s63_si -= s63_si.min()
s63_si /= s63_si.max()
s63_si = (1 - s63_si)*10

fp = findpeaks(method='topology', limit=1)
results = fp.fit(s63_si, x=s63_wl)
plt.scatter(s63_wl, results['Xdetect'])
plt.plot(s63_wl, s63_si)
# %%
# Load library
import numpy as np
from findpeaks import findpeaks

# Data
i = 10000
xs = np.linspace(0,3.7*np.pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Initialize
fp = findpeaks()
results = fp.fit(X)
# Plot
fp.plot1d()

fp = findpeaks(method='topology', limit=1)
results = fp.fit(X)
fp.plot1d()
fp.plot_persistence()
# %%
