#%%
from astropy.utils.diff import difflib
import matplotlib.pyplot as plt 
import numpy as np 
import astropy.io.fits as fits
from glob import glob
# %%

dir = '../data/hms1_Oct10Aurora/20241010/*.fit*'
fnames = glob(dir)
print(len(fnames))

#%%
with fits.open('/home/charmi/Projects/hitmis_analysis/data/hms1_Oct10Aurora//20241010/163944.474.fits') as hdul:
    data = hdul[1].data
    header = hdul[1].header
# %%
vmin = np.percentile(data,1)
vmax = np.percentile(data,99)
plt.imshow(data, vmin=vmin,vmax = vmax)
plt.colorbar()

for i, hdu in enumerate(hdul):
    # Check if the keyword 'EXPOSURE' is in the header
    if 'EXPOSURE' in hdu.header:
        value = hdu.header['EXPOSURE']
        print(f'HDU {i}: EXPOSURE = {value}')
# %%


comments = np.asarray(header.comments)
expstr = np.where(comments == 'Exposure time (ns)')[0][0]
# %%
header[expstr]
# %%
expstr
# %%
np.where(comments == 'Exposure time (ns)')[0][0]
# %%
header
# %%
fn = '/home/charmi/Projects/hitmis_analysis/data/hms1_Oct10Aurora//20241010/163944.474.fits'
with fits.open(fn) as hdul:
    header = hdul[1].header
    expstr = difflib.get_close_matches('EXPOSURE', list(header))[0]
if '_US' in expstr: TO_SEC = 1e-6
elif '_MS' in expstr: TO_SEC = 1e-3
else: 
    hdrcomm = np.asarray(header.comments)
    expstr = np.where(hdrcomm == 'Exposure time (ns)')[0][0]
    TO_SEC = 1e-9 #nm -> s
expstr# %%

# %%
