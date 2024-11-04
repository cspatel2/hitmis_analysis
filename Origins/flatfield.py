# %%
import astropy.io.fits as pf
from glob import glob
import xarray as xr
import numpy as np
# %%

files = glob('flat/*/*.fit*')
# %%
out = None
for file in files:
    with pf.open(file) as hdu:
        if out is None:
            out = hdu[0].data.astype(float)
        else:
            out += hdu[0].data.astype(float)
    
# %%
out = np.asarray(out)
# %%
out /= out.max()
# %%
flat = pf.ImageHDU(out, header = pf.Header([pf.Card('Gain', 0, 'dB'),
                                     pf.Card('CAMERA', 'ZWO ASI533MM'),
                                     pf.Card('INSTR', 'HiT&MIS A'),
                                     pf.Card('SLIT', '50um 25mm sep')]))
flat.writeto('hmsa_50um_25mm_asi533_gain0.fit', overwrite=True)
# %%
flat
# %%
flat.header
# %%
