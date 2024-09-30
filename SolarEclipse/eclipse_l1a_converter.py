#%% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

#%%
import matplotlib.pyplot as plt
import numpy as np 
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from hmspython.Utils import _files
from hmspython.Utils import _Utility
from glob import glob
import astropy.io.fits as fits
import xarray as xr

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
mapping = MapPixel2Wl(predictor)
#%%
imgs,tstamps,time,exposure,gain,camtemp = []
with fits.open(fn) as hdul:
    header = hdul[1].header
    tstamp = int(header['HIERARCH TIMESTAMP']) #ms
    time =  _files.time_from_tstamp(tstamp) #datetime object
    exposure = int(header['HIERARCH EXPOSURE_MS']) #ms
    gain = header['GCOUNT']
    camtemp = float(header['CCDTEMP']) #degrees Celcius
    data = hdul[1].data
    data = _files.open_eclipse_data(data) #crop and resize img to 3008x3008
    data =data/(exposure*1e-3) # [ADU/s]
#%%

img_data = data
#%%
s = mapping.straighten_img(wavelength=486.1, img=img_data, rotate_deg=1.25)


# %%
simg,img,wlaxis = s
pix_y = np.arange(np.shape(img)[0])
# %%
ds = xr.Dataset(
    data_vars=dict(
        img=(["tstamp","wavelength", "pix_y"], data),
    ),
    coords=dict(
        tstamp = ("tstamp",tstamp),
        wavelength = ("wavelength", wlaxis),
        pix_y = ("pix_y", pix_y),
        gain = ('tstamp',gain),
        exposure = ('tstamp', exposure),
        camtemp = ('tstamp', camtemp),
        time = ('tstamp', time )


    ),
    attrs=dict(description="Weather related data."),
)
# %%
