#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from hitmis_Instrument.img_predictor import HMS_ImagePredictor, load_pickle_file
import astropy.io.fits as pf
from skimage import exposure
from glob import glob



# %%

# %%
imgdir = '/home/charmi/Projects/hitmis_analysis/data/hms1_EclipseRaw/EclipseDay/20240408/*fit'
fnames = glob(imgdir)
len(fnames)

# %%
idx = 1700
impath = fnames[idx]
if '.png' in impath:
    data = plt.imread(impath)
elif '.fit' in impath: 
    with pf.open(impath) as hdul:
        data = hdul[1].data.astype(np.float64)
else:
    raise ValueError("Unsupported file format")
# data = self.clip_image_percentiles(data)
plt.imshow(data)
#%%

#%%

#%%



# %%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
#%%
predictor = HMS_ImagePredictor('a',67.3,50, 90-0.3)
#%%
# data = exposure.equalize_hist(data)
img_data = data


# Crop the image
# Remove 100 rows from the bottom and 10 columns from the right
cropped_img_data = img_data[:-95, :-95]

# Resize the cropped image to 3008x3008 pixels
imglen = 3008
zoom_factor = (imglen / cropped_img_data.shape[0], imglen/ cropped_img_data.shape[1])
resized_img_data = zoom(cropped_img_data, zoom_factor, order=3)  # order=3 for cubic interpolation

# Reverse the image along both axes
reversed_img_data = np.flip(resized_img_data, axis=(0, 1))

# # Save the reversed and resized image as a new FITS file
# hdu = fits.PrimaryHDU(reversed_img_data)
# hdul_new = fits.HDUList([hdu])
# output_fits_path = 'path_to_save_resized_image.fits'
# hdul_new.writeto(output_fits_path, overwrite=True)

# Optionally, display the image using matplotlib
fig = predictor.plot_spectral_lines(ImageAt='detector', Tape2Grating=True,fprime = 430.798)
plt.imshow(reversed_img_data,cmap = 'viridis')
plt.colorbar()
plt.show()


# %%
predictor.fit_fprime('hydrogen', [67.6,67.4])
# %%
predictor.fit_fprime('oxygen', [67.7,67.55])

# %%
