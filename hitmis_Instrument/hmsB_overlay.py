#%%
import numpy as np
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
import matplotlib.pyplot as plt
from glob import glob
# %%


#%%
imgdir = 'Images/hmsB_hlamp_08232024/*.png'
fnames = glob(imgdir)
# %%
predictor = HMS_ImagePredictor(hmsVersion='b',mgammadeg=90+0.6,alpha=68)
# predictor.overlay_on_Image(fnames[0],vmin = 0.5, vmax = 1.5)
# %%
for i in range(len(fnames)):
    predictor.overlay_on_Image(fnames[i],vmin = 0.5, vmax = 1.5)

# %%
predictor.fit_fprime('hydrogen')
# %%
predictor.overlay_on_Image(fnames[-1],vmin = 0.5, vmax = 1.5)
# %%
predictor.overlay_on_Image(fnames[0],vmin = 0.5, vmax = 1.5)
# %%
fnames[-1]
# %%
predictor.overlay_on_Image(fnames[0], [486.1, 427.8, 557.7, 630, 656.3, 777.4, 720.0, 740.0])
# %%
predictor.overlay_on_Image(fnames[0], [486.1, 427.8, 557.7, 630, 656.3, 777.4,690])

# %%
predictor = HMS_ImagePredictor(hmsVersion='a',mgammadeg=90+0.6,alpha=67.8)
#%%
predictor.plot_spectral_lines('Detector',True,wls = [486.1, 427.8, 557.7, 630, 656.3, 777.4,720,740])

# %%
for i in np.arange(65,70,0.25):
    predictor = HMS_ImagePredictor(hmsVersion='b',mgammadeg=90+0.6,alpha=i)
    predictor.plot_spectral_lines('Detector',True,wls = [486.1, 427.8, 557.7, 630, 656.3, 777.4,720,740, 784, 786])
# %%
for i in np.arange(65,70,0.25):
    predictor = HMS_ImagePredictor(hmsVersion='b',mgammadeg=90+0.6,alpha=i)
    predictor.plot_spectral_lines('Detector',True,wls = [486.1, 427.8, 557.7, 630, 656.3, 777.4,720,740, 780, 788,480., ])
# %%

predictor = HMS_ImagePredictor(hmsVersion='b',mgammadeg=90+0.6,alpha=68.25)
predictor.plot_spectral_lines('Detector',True,wls =  [486.1, 427.8, 557.7, 630, 777.4,720,740,780 ,788,478 ])
# %%

predictor = HMS_ImagePredictor(hmsVersion='bo',mgammadeg=90,alpha=66.25)


# %%
img = predictor.plot_spectral_lines('Mosaic',True,wls = [557.7, 630.0,427.8, 784.1,777.4,486.1,656.3 ,654.4])
plt.axhline(y = 578.5)
plt.axvline(x = 404.75, ymin = 0, ymax = 0.5) #left edge
plt.axvline(x = 382, ymin = 0, ymax = 0.5)
plt.axvline(x = 390, ymin = 0.5, ymax = 1)
plt.axvline(x = 383, ymin = 0.5, ymax = 1)
plt.axvline(x = 378, ymin = 0.5, ymax = 1)
plt.axvline(x = 363, ymin = 0.5, ymax = 1)






# %%
img = predictor.plot_spectral_lines('Mosaicwindow',True,wls = [557.7, 630.0,427.8, 784.1,777.4,486.1,656.3])

# %%
