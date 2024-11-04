# %%
import os
import sys
import wave
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import astropy.io.fits as pf
from hmspython.Utils._files import *
from hmspython.Utils._Utility import *
from hmsdesigner._ImgPredictor import HMS_ImagePredictor # type: ignore
from hmsdesigner._Pixel2wlMapping import MapPixel2Wl # type: ignore
from tqdm import tqdm # type: ignore
from scipy.ndimage import gaussian_filter1d # type: ignore
import xarray
import scipy.signal as signal
import scipy.optimize as optimize
from skmpython.GenericFit import GenericFitFunc,GenericFitManager





#%% TEST SLIT 
config = '/home/charmi/Projects/Hitmis/hmsdesigner/configs/hmsa_aurora_slittest.toml'
predictor = HMS_ImagePredictor(config)
# %%
fnames = glob('slittest_data_raw/hlamp/2024-10-24_16_48_08Z/*.fit*')
impath = fnames[1]
with pf.open(impath) as hdul:
    data = hdul[0].data.astype(np.float64)
    

 # %%
img = predictor.plot_spectral_lines('Detector', True, wls=[557.7, 630.0, 427.8, 777.4, 486.1, 656.3, 486.1], mosaic=True, measurement=True,fprime=442.7)
vmin = np.percentile(data,1)
vmax = np.percentile(data,99)
plt.imshow(data, cmap = 'pink', vmin = vmin, vmax = vmax)
plt.colorbar()

# %%
#Inititalize mapping
mapping = MapPixel2Wl(predictor)
#%%
#Straighten img
wl = 557.7
simg,img,wlaxis = mapping.straighten_img(wavelength = wl, img = data, rotate_deg = -0.2) 
#%%
#plot a bigger version of straightened img
vmin = np.nanpercentile(simg,1)
vmax = np.nanpercentile(simg,99)


plt.imshow(simg,vmin = vmin,vmax = vmax, aspect = 'auto')
plt.axvline(find_nearest(wlaxis,wl)[0], color = 'white', linewidth = 0.5)
plt.colorbar()
plt.title(f'{wl} nm')
# %%
##############################################################
def minmax(arr: np.ndarray)->np.ndarray:
    arr = arr.copy()
    arr -= np.nanmin(arr)
    arr /= np.nanmax(arr)
    return arr

# %%
wl = 427.8
sounce = 'Neonlamp'
wlstr = str(int(wl*10))
fnames = glob(f'slittest_data_l1a/{sounce}*{wlstr}.nc')
ds  = xarray.open_dataset(fnames[0])
# %%
img = ds['img'][1]
img.plot.imshow(origin = 'upper',cmap = 'rainbow')
# plt.xlim(486,487)
# plt.axvline(wl, color = 'white', linewidth = 0.5, linestyle = '--')
# %%
# wl = 486.1
# peakwl = wl +.2
peakwl = wl -0.017
wlmin = peakwl - .5
wlmax = peakwl + .5
line = img[1400:1420].sel(wavelength = slice(wlmin,wlmax)).sum(axis = 0)

# %%
line.plot()
plt.axvline(peakwl, color = 'orange', linewidth = 0.5, linestyle = '--')

# %%
wlaxis = line.wavelength.values
counts = line.values 
counts = minmax(counts) #range 0 - 1
# %%
# fline = gaussian_filter1d(counts,50) #smooth
# fline = line
# popt = np.polyfit(wlaxis, fline, 1) #fit background
# plt.plot(wlaxis,counts) #plot data
# plt.plot(wlaxis,fline) # plot smooth
# plt.plot(wlaxis, popt[0]*wlaxis + popt[1]) # plot fit
# plt.plot(wlaxis, counts - (popt[0]*wlaxis + popt[1]))
# %%
# counts = counts - (popt[0]*wlaxis + popt[1])
counts -= np.min(counts) # change origin to x = 0
plt.plot(wlaxis,counts)

# %% Delta Function at the spectral line
N = len(wlaxis)
dwlaxis = np.linspace(wlmin,wlmax,N)
delta = signal.unit_impulse(len(wlaxis),'mid')
plt.plot(dwlaxis,delta)
plt.title('Delta function')
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')


#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wlaxis, counts)
ax.plot(dwlaxis, delta)
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')
# %%
# ccounts = np.interp(dwlaxis, wlaxis, counts)
counts = counts # O(λ)
fcounts = np.fft.rfft(counts) # O(v), v = 1/λ
fdelta = np.fft.rfft(delta) # K(v), v = 1/λ
finstr = fcounts / fdelta # I(v), v = 1/λ
instr = np.fft.irfft(finstr)
instr = np.fft.fftshift(instr) # I(λ) 
# instr = np.fft.fftshift(np.abs(np.fft.irfft(finstr))) # I(λ) 

# %%

# plt.plot(swl - swl.mean(), np.fft.fftshift(np.abs(instr)))
plt.plot(wlaxis[:len(instr)],instr )
# plt.axvline(wl)
# plt.xlim(0, 0.1)
plt.title('Instrument Function')
plt.xlabel('λ (nm)')
# %%
#Reverse the process to check if this instrument function is right
recounts = np.convolve(delta,instr, 'same')
plt.plot(wlaxis,counts, label = 'Measured')
plt.plot(wlaxis,recounts, label = 'Convolved')
plt.legend(loc = 'best')
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')
# %% fit a gaussian 
# wlaxis = wlaxis = np.mean(wlaxis)
def gaussian(x, xo, a, c):
    return a * np.exp(-(x - xo)**2 / (2 * c**2))

# Fit the data
popt, pcov = optimize.curve_fit(gaussian, wlaxis, instr, p0=[peakwl, 2, 2])
#%%

fwhm = 2*np.sqrt(2*np.log(2)) * popt[-1]
plt.figure(figsize = (7,4))
plt.plot(wlaxis,instr, label = 'instr')
plt.plot(wlaxis,gaussian(wlaxis,*popt), label = f'gaussian fit,\n FWHM = {fwhm:0.4f} nm')
plt.legend(loc = 'best')
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')
plt.title('Instrument Func using Lamp')


# %%
#Test the instrument function above with solar spectra
#1. Instrument function = I(λ), from above
#2. sky spectra = O(λ)
#3. solar spectra = K(λ)
#4. test O(λ) = convolve( solar spectra, instrument function)
#5. plot test spectr and sky spectra together to see if they match

#%%
#%% 2. Solar Spectra 
solspec = xarray.open_dataset('../solar_spectra_air.nc')
solspec = solspec.sel(wavelength = slice(wlmin, wlmax))

##### Match and plot ####
wlsol = solspec.wavelength.values
csol = minmax(solspec.irradiance.values)
#%%  3. get sky spectra

sounce = 'Sky'
wlstr = str(int(wl*10))
fnames = glob(f'slittest_data_l1a/{sounce}*{wlstr}.nc')
ds  = xarray.open_dataset(fnames[0])

img = ds['img'][1]
img.plot.imshow(origin = 'upper')
plt.axvline(wl, color = 'white', linewidth = 0.5, linestyle = '--')

plt.figure()
wlmin = wl - .5
wlmax = wl + .5
line = img[1200:1250].sel(wavelength = slice(wlmin,wlmax)).sum(axis = 0)
line.plot()
plt.axvline(wl, color = 'orange', linewidth = 0.5, linestyle = '--')

wlsky = line.wavelength.values
csky = line.values 
# csky = minmax(counts) #range 0 - 1

fline = gaussian_filter1d(csky,50)
popt = np.polyfit(wlaxis, csky, 1)
plt.plot(wlsky,csky)
plt.plot(wlsky,fline)
plt.plot(wlsky, popt[0]*wlaxis + popt[1])
#%%
plt.plot(wlsky, csky - (popt[0]*wlaxis + popt[1]))

csky = csky - (popt[0]*wlaxis + popt[1]) # final, background subtracted, normalized

## interpolated to match the size of the solar data
csky = minmax(np.interp(wlsol,wlsky,csky))
wlsky = wlsol


#%% Match sky peak to sol peak

#peak at zero and then readujst to 486.1
wlsol_n = wlsol - wlsol[np.argmin(csol)] + 486.1
wlsky_n = wlsky - wlsky[np.argmin(csky)] + 486.1

plt.figure(figsize = (7,4))
plt.plot(wlsol_n,csol, label = 'Solar')
plt.plot(wlsky_n,csky, label = 'sky-Measured')
plt.legend(loc = 'best')
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')

#%%
# 4. test O(λ) = convolve( solar spectra, instrument function)
# instr = np.interp(wlsol,wlsky,instr)
recounts = minmax(np.convolve(csol,instr, 'same'))

# #%% interp our data to match wl of csol
# csky_interp = minmax(np.interp(wlsol,wlsky,csky))
#%%
#5. plot test spectr and sky spectra together to see if they match
plt.figure(figsize = (7,4))
# plt.plot(wlsol_n,csol, label = 'Solar')
plt.plot(wlsky_n,csky, label = 'sky-Measured')
plt.plot(wlsol_n,recounts, label = 'Convolved-(solar X instr)')


plt.legend(loc = 'best')
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')

#%%
# # %%
fcounts = np.fft.rfft(csky)
fsol = np.fft.rfft(csol)
finstr = fcounts / fsol
instr = np.fft.irfft(finstr)
instr = minmax(np.fft.fftshift(np.abs(instr)))
minlen = min(len(instr), len(wlsol_n))
instr = instr[:minlen]
swl = wlsol_n[:minlen]
swl = swl - swl.mean()
# %%
ofset = 0.01
mask = (((swl.mean()-ofset)<= swl) &((swl.mean()+ofset)>= swl))
idx = np.where(mask)
fitwl = swl[idx]
fitinstr = instr[idx]
plt.plot(fitwl, fitinstr)


# plt.xlim(-0.01, 0.01)

#%%
# def genfit():
#     a * np.exp(-(x - xo)**2 / (2 * c**2)) + 
    
# %%
def fitfunc(x,a0,a1,a3,c):
    return  a0 + a1 *x + a3 * np.exp(-(x)**2 / (2 * c**2))
# Fit the data
popt, pcov = optimize.curve_fit(fitfunc,fitwl ,fitinstr,
                                p0=[ 0.1,0.01,1, 0.004])
#%%

fwhm = 2*np.sqrt(2*np.log(2)) * np.abs(popt[-1])
plt.figure(figsize = (7,4))
plt.plot(fitwl,fitinstr, label = 'instr')
plt.plot(fitwl,fitfunc(fitwl,*popt), label = f'gaussian fit,\n FWHM = {fwhm:0.4f} nm')
plt.legend(loc = 'best')
plt.xlabel('λ (nm)')
plt.ylabel('Normalized Intensity')
plt.title('Instrument Func using Sky Spectra')

#%%
# x = fitwl
# y = fitinstr

# p0 = (0, 0.1, 0, 0, 1, 0.0025)  # initial guess parameters
# p_low = (0, 0, 0, -1, 0, 0)  # initial guess lower bounds
# # inital guess upper bounds
# p_high = (1, 1, 1, 1, 1, 0.0075)
# def background_fcn(x, x0, a0, a1): return a0 + \
#     a1 * (x-x0) 
# # create the generic fit function and register the background
# bfuncs = GenericFitFunc(
#     background_fcn=background_fcn, num_background_params=3)

# def feature_fcn(x, c, a, w): return a*np.exp(-((x-c)/(w**2)))
# bfuncs.register_feature_fcn(
#     fcn=feature_fcn, num_params=3)  # register the feature
# bfuncs.finalize()  # finalize the registration

# # create the fit manager
# gfit = GenericFitManager(
#     x, y, p0=p0, baseclass=bfuncs, window_title='Test')
# print('GenericFitManager using %s backend.' %
#         (gfit.baseclass_name))  # print the base class
# # print('Original:', p_def)  # print the original parameters
# gfit.run(ioff=False, close_after=False, bounds=(
#     p_low, p_high), p0=p0)  # run the fit
# print('Derived:', gfit.param)  # print the derived parameters
# # print the mean squared error and noise
# # print('Error:', gfit.meansq_error, ', Noise:', noise.std(),
#         # ', Initial guess error:', gfit.integrated_err())
# print(gfit.wrap_results())
# print('Number of iterations:', gfit.iterations)
# gfit.plot()  # plot the fit result
# # %%
# gfit.param
# # %%
# fwhm = 2*np.sqrt(2*np.log(2)) * popt[-1]
# # %%
# fwhm
# %%
