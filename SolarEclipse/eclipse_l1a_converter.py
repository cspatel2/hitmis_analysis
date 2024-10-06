#%% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

#%%
import argparse
from datetime import datetime
import lzma
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np 
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Diffraction._Pixel2wlMapping import MapPixel2Wl
from hmspython.Utils import _files
from hmspython.Utils import _Utility
from hmspython.Utils import _time
from glob import glob
import astropy.io.fits as fits
import xarray as xr
from tqdm import tqdm
#%% 
# %% Paths
# rootdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_EclipseRaw/EclipseDay/'
PATH = os.path.dirname(os.path.realpath(__file__))
IMGSIZE = 1024
# %% Argument Parser
parser = argparse.ArgumentParser(
    description='Convert HiT&MIS L0 data to L1 data, with exposure normalization and Dark subtraction. It  using HMS_ImgPredictor()(hmspython.Diffraction._ImgPredictor) and  MapPixel2WL() (from hmspython.Diffraction._Pixel2wlMapping) to extact ROI and performs line straightening. This program will not work without hmsParams.pkl and hmsWlParams.pkl present.')
# %% Add arguments
parser.add_argument('rootdir',
                    metavar='rootdir',
                    type=str,
                    help='Root directory containing HiT&MIS data')
# parser.add_argument('dest',
#                     nargs='?',
#                     default=os.getcwd(),
#                     help='Root directory where L1 data will be stored')
parser.add_argument('dest_prefix',
                    nargs='?',
                    default=os.getcwd(),
                    help='Prefix of final L1 data file name')
parser.add_argument('--wl',
                    required=False,
                    type=float,
                    help='Wavelength to process')


# %% Get all subdirs
def list_all_dirs(root):
    flist = os.listdir(root)
    # print(flist)
    out = []
    subdirFound = False
    for f in flist:
        if os.path.isdir(root + '/' + f):
            subdirFound = True
            out += list_all_dirs(root + '/' + f)
    if not subdirFound:
        out.append(root)
    return out


def getctime(fname):
    words = fname.rstrip('.fit').split('_')
    return int(words[-1])

#%%
args = parser.parse_args()

# 1. Get files ##################################################3

# Get Root directory
rootdir = args.rootdir
if not os.path.isdir(rootdir):
    print('Specified root directory for L0 data does not exist.')
    sys.exit()
#Get list of all directories within root directory
dirlist = list_all_dirs(rootdir)  
#get list of .fit files in all directories sorted by ucttime 
filelist = []
for d in dirlist:
    if d is not None:
        f = glob(d+'/*.fit')
        f.sort(key=getctime)
        filelist.append(f)
#flatten list of files
flat_filelist = []
for f in filelist:
    for img in f:
        flat_filelist.append(img)
        
print(f'Number of fits files to process: {len(flat_filelist)}')
if len(flat_filelist) == 0:
    raise ValueError('No .fit files in provided rootdir.')

flist = flat_filelist
flist.sort(key=getctime)
# Get timeframe
start_date = datetime.fromtimestamp(getctime(flist[0])*0.001)
end_date = datetime.fromtimestamp(getctime(flist[-1])*0.001)
print('First image:', start_date)
print('Last image:', end_date)
print('\n')
#############################################################

# 2. Initialize ############################################

#intialize model
predictor = HMS_ImagePredictor('ae',67.32,50, 90-0.45,IMGSIZE)

#Initialize wl 
wl = args.wl
    #check that its a valid wl for this mosaic
wl_str = str(int(wl*10))
wlarr_str = _Utility.flatten_list(predictor.hmsParamDict['MosaicFilters']) #wl in str(Angstrom)
wlarr_nm = [int(x)/10 for x in wlarr_str] #wl in float(nm)

if wl_str not in wlarr_str:#wl in str(Angstrom)
    raise ValueError(f'Invalid wl, must be one of the following: {wlarr_nm}')
print(f'Processing ROI: {wl} nm')

#Inititalize wlmapper
mapping = MapPixel2Wl(predictor)

#Initialize Dark data dict
darkPATH = '../hitmis_pipeline/pixis_dark_bias.xz'
with lzma.open(darkPATH, 'rb') as dfile:
    darkDict = pickle.load(dfile)

#Initialize varirable arrays to fill
tstamps,times,exposures,gains,camtemps = [],[],[],[],[]
simgs = [] # straightened imgs

# 3. Process Images 
for fidx,fn in enumerate(tqdm(flist)):
    with fits.open(fn) as hdul:
        header = hdul[1].header
        tstamp = int(header['HIERARCH TIMESTAMP']) #ms
        tstamps.append(tstamp)
        times.append(_time.time_from_tstamp(tstamp))#datetime object
        exposure = int(header['HIERARCH EXPOSURE_MS'])*1e-3 #ms -> s
        exposures.append(exposure) 
        gains.append(header['GCOUNT'])
        camtemps.append(float(header['CCDTEMP'])) #degrees Celcius
        
        #1. get img
        data = np.asarray(hdul[1].data,dtype = float) #counts
        #2. dark/bais correct img
        data -= darkDict['bias'] + (darkDict['dark']*exposure) #counts
        #3. crop and resize img to IMGSIZE
        data = _files.open_eclipse_data(data,imgsize=IMGSIZE) #counts
        #4. total counts -> counts/sec
        data = data/exposure
        #5. straighten ROI within img
        simg,img,wlax = mapping.straighten_img(wavelength=wl, img=data, rotate_deg=1.25,plot=False)
        simgs.append(simg) #counts; shape(fidx, IMGSIZE, IMGSIZE)
        #6. Save wlaxis (reference axis that the img is straighted to.)
        if fidx == 0: 
            wlaxis = wlax
            pix_y = np.arange(np.shape(simg)[0]) 


# 4. Create Dataset and Save
createdtime = datetime.now().strftime('%a %d %b %Y, %I:%M%p')
attr_time = str(createdtime) + ' EST'
ds = xr.Dataset(
    data_vars=dict(
        img=(("tstamp","pix_y","wavelength"), np.asarray(simgs,dtype=float),
             {'Description': 'Dark-subtracted straightened images',
              'units': 'ADU/nm/s'
                   })
    ),
    coords=dict(
        tstamp = ("tstamp",np.asarray(tstamps,dtype = int),
                  {'Description': 'timestamp',
                   'units': 's'
                   }),
        wavelength = ("wavelength", np.asarray(wlaxis, dtype =float),
                  {'Description': 'Reference wavelength [x]axis that img is straighted with.',
                   'units': 'nm'
                   }),
        pix_y = ("pix_y", np.asarray(pix_y,dtype=int),
                  {'Description': 'Detector Pixel Position in Y [rows] ',
                   'units': 'Pixel Postion Y'
                   }),
        gain = ('tstamp',np.asarray(gains,dtype=int),
                  {'Description': 'Camera gain',
                   'units': 'electrons/ADU'
                   }),
        exposure = ('tstamp', np.asarray(exposures, dtype=float),
                  {'Description': 'Exposure time of img',
                   'units': 's'
                   }),
        camtemp = ('tstamp', np.asarray(camtemps,dtype=int),
                  {'Description': 'Camera temperature',
                   'units': 'Degree Celsius'
                   }),
        # time = ('tstamp', np.asarray(times,dtype='datetime64[ns]'),
        #           {'Description': 'Time as datetime',
        #            'units': 'UTC'
        #            })
    ),
    attrs=dict(Description="HMS A - Straightened Eclipse data.",
               ROI = f'{str(wl)} nm',
               CreationDate =  attr_time
))

# destdir = args.dest
prefix = args.dest_prefix

yymmdd = int(start_date.strftime("%Y%m%d"))

outfname = f"{prefix}_{yymmdd}_{wl_str}.nc"

print('Saving %s...\t' % (outfname), end='')
sys.stdout.flush()

ds.to_netcdf(outfname)
print('Done.')

#%%
ds = xr.open_dataset('Eclipse_20240408_5577.nc')
# # %%
# ds
# %%
