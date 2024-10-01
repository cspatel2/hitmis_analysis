#%% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

#%%
from sqlite3 import Timestamp
import matplotlib.pyplot as plt
import numpy as np 
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from hmspython.Utils import _files
from hmspython.Utils import _Utility
from glob import glob
import astropy.io.fits as fits
import xarray as xr
from tqdm import tqdm
# %%
#for each file 

# 1. open .fits file
    #collect img, exposure, timestamp,gain
# 2. straighten img
# 3. store in a ds that can be saved as 
#%% 
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_SolarSpectra/20240507/*.fit'
fnames = glob(imgdir)
print(len(fnames))
fn = fnames[50]
#%%
# #%%
predictor = HMS_ImagePredictor('ae',67.32,50, 90-0.45)
#%%
mapping = MapPixel2Wl(predictor)
#%%

#Inititalize parameters
tstamps,times,exposures,gains,camtemps = [],[],[],[],[]

wls_str = _Utility.flatten_list(predictor.hmsParamDict['MosaicFilters']) #wl in str(Angstrom)

imgDict = {el:[] for el in wls_str} #dict of straighted imgs
wlaxisDict = {el:[] for el in wls_str} #dict of wlaxis

wl_iter = [int(x)/10 for x in wls_str] #wl in nm
#%%
for fidx,fn in enumerate(fnames[:5]):
    with fits.open(fn) as hdul:
        header = hdul[1].header
        tstamp = int(header['HIERARCH TIMESTAMP']) #ms
        tstamps.append(tstamp)
        times.append(_files.time_from_tstamp(tstamp))#datetime object
        exposure = int(header['HIERARCH EXPOSURE_MS'])
        exposures.append(exposure) #ms
        gains.append(header['GCOUNT'])
        camtemps.append(float(header['CCDTEMP'])) #degrees Celcius
        data = hdul[1].data
        data = _files.open_eclipse_data(data) #crop and resize img to 3008x3008
        data =data/(exposure*1e-3) # [ADU/s]
        for wl in tqdm(wl_iter):
            simg,img,wlaxis = mapping.straighten_img(wavelength=wl, img=data, rotate_deg=1.25,plot=False)
            wlstr = str(int(wl*10))
            imgDict[wlstr].append(simg)
            if fidx == 0: wlaxisDict[wlstr] += list(wlaxis)

            
        
        
        
#%%
# %%
ds = xr.Dataset(
    data_vars=dict(
        img=(["tstamp","wavelength", "pix_y"], data),
    ),
    coords=dict(
        tstamp = ("tstamp",tstamp),
        wavelength = ("wavelength", wlaxis),
        pix_y = ("pix_y", pix_y),
        gain = ('tstamp',gains),
        exposure = ('tstamp', exposures),
        camtemp = ('tstamp', camtemps),
        time = ('tstamp', times )


    ),
    attrs=dict(description="HMS A - Eclipse data."),
)
# %%
a = ['a', 'b', 'c']
tdict = {el:[] for el in a}
# %%


# %%
import concurrent.futures
import xarray as xr
from astropy.io import fits
from tqdm import tqdm

# Initialize parameters
tstamps, times, exposures, gains, camtemps = [], [], [], [], []

wls_str = _Utility.flatten_list(predictor.hmsParamDict['MosaicFilters'])  # wl in str(Angstrom)
wl_iter = [int(x)/10 for x in wls_str]  # wl in nm

# Function to process a single panel
def process_panel(panel_idx, wl, data, exposure):
    simg, img, wlaxis = mapping.straighten_img(wavelength=wl, img=data, rotate_deg=1.25, plot=False)
    wlstr = str(int(wl * 10))
    return wlstr, simg, wlaxis

# Function to save data to NetCDF
def save_to_netcdf(wlstr, data, wlaxis, tstamp, gains, exposures, camtemps, times):
    ds = xr.Dataset(
        data_vars=dict(
            img=(["tstamp", "pix_y"], data),
        ),
        coords=dict(
            tstamp=("tstamp", tstamp),
            wavelength=("wavelength", wlaxis),
            pix_y=("pix_y", range(data.shape[1])),
            gain=('tstamp', gains),
            exposure=('tstamp', exposures),
            camtemp=('tstamp', camtemps),
            time=('tstamp', times)
        ),
        attrs=dict(description="HMS A - Eclipse data.")
    )
    ds.to_netcdf(f'panel_{wlstr}.nc')

# Main processing loop
for fidx, fn in enumerate(fnames[:5]):
    with fits.open(fn) as hdul:
        header = hdul[1].header
        tstamp = int(header['HIERARCH TIMESTAMP'])  # ms
        tstamps.append(tstamp)
        times.append(_files.time_from_tstamp(tstamp))  # datetime object
        exposure = int(header['HIERARCH EXPOSURE_MS'])
        exposures.append(exposure)  # ms
        gains.append(header['GCOUNT'])
        camtemps.append(float(header['CCDTEMP']))  # degrees Celsius
        data = hdul[1].data
        data = _files.open_eclipse_data(data)  # crop and resize img to 3008x3008
        data = data / (exposure * 1e-3)  # [ADU/s]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for wl in wl_iter:
                futures.append(executor.submit(process_panel, panel_idx, wl, data, exposure))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                wlstr, simg, wlaxis = future.result()
                imgDict[wlstr].append(simg)
                if fidx == 0:
                    wlaxisDict[wlstr] = wlaxis

# Save each panel to a NetCDF file
for wlstr, img_list in imgDict.items():
    data = np.stack(img_list)  # Stack images along a new axis
    save_to_netcdf(wlstr, data, wlaxisDict[wlstr], tstamps, gains, exposures, camtemps, times)
