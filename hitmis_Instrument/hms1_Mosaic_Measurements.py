#%%
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
# %%
# imgdir = 'Images/hmsA_testimg/img.png'
# %%
def annotate_width(x_offset_mm: float, len_mm: float, toppanel: bool):
    axis = 0
    # Set the starting y position based on the panel selection
    if toppanel: y_base = 750
    else: y_base = 2250

    # Calculate the pixel positions for x-offset and length
    if x_offset_mm == 0:x_offset_pix = 0
    else:x_offset_pix = predictor.mm2pix(x_offset_mm, axis, predictor.pix)
    
    len_pix = predictor.mm2pix(len_mm, axis, predictor.pix)

    xy = (x_offset_pix,y_base)
    xytext = (x_offset_pix+ len_pix,y_base)
    
    # Annotate with a slight y offset increment to prevent overlapping
    plt.annotate('', xy=xy, xytext=xytext, 
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    
    # Position the text in the middle of the line with the current y-offset
    plt.text(x_offset_pix + len_pix / 2, y_base, f'{len_mm:.2f} mm',
            color='black', fontsize=7, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.3'))

def annotate_height(y_offset_mm: float, len_mm: float, toppanel: bool):
    axis = 1
    x_base = 100
 
    # Calculate the pixel positions for x-offset and length
    if y_offset_mm == 0:y_offset_pix = 0
    else:y_offset_pix = predictor.mm2pix(y_offset_mm, axis, predictor.pix)
    
    len_pix = predictor.mm2pix(len_mm, axis, predictor.pix)

    xy = (x_base,y_offset_pix)
    xytext = (x_base,y_offset_pix+ len_pix)
    
    # Annotate with a slight y offset increment to prevent overlapping
    plt.annotate('', xy=xy, xytext=xytext, 
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    
    # Position the text in the middle of the line with the current y-offset
    plt.text(x_base,y_offset_pix + len_pix / 4, f'{len_mm:.2f} mm',
            color='black', fontsize=8, ha='center', va='center',rotation = 270,
            bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.3'))
# %%

y_offset = 0 
predictor = HMS_ImagePredictor(hmsVersion='a',alpha=67.39)
imgdir = 'Images/hmsA_testimg/ccdi_20240811_003321.fits'
predictor.overlay_on_Image(imgdir)

m6563 = 20.00
m4861 = predictor.MosaicWindowWidthmm-m6563

m4278 = 17.5
m6300 = 27.20 - m4278
m5577 = 37.3 - m6300 - m4278
m7774 = predictor.MosaicWindowWidthmm-m5577 - m6300 - m4278

plt.axhline(predictor.g0,color ='red') #half of slit len
plt.axvline(predictor.mm2pix(m6563,0,predictor.pix), 0.5,1,color = 'cyan')
annotate_width(0,m6563,True)
annotate_width(m6563,m4861,True)

y_offset = 0
plt.axvline(predictor.mm2pix(m4278,0,predictor.pix), 0,0.5, color = 'blue')
annotate_width(0,m4278,False)
plt.axvline(predictor.mm2pix(m4278+m6300,0,predictor.pix), 0,0.5, color = 'red')
annotate_width(m4278,m6300,False)
plt.axvline(predictor.mm2pix(m4278+m6300+m5577,0,predictor.pix), 0,0.5, color = 'green')
annotate_width(m4278+m6300,m5577,False)
annotate_width(m4278+m6300+m5577,m7774,False)

annotate_height(0,predictor.MosaicWindowHeightmm/2,True)
annotate_height(predictor.MosaicWindowHeightmm/2,predictor.MosaicWindowHeightmm/2,True)

plt.savefig("Panel_measuments_atMosaicWindow.png")



# %%
Mosaic_ParamDict={
    '7774':{
        'PanelLetter':'a',
        'PanelWindowWidthmm':12.22 ,
        'PanelWidthmm':12.22,
        'PanelWindowHeightmm':25.02,
        'PanelHeightmm':25.02+2.9,
    },
    '5577':{
        'PanelLetter':'b',
        'PanelWindowWidthmm':10.10 ,
        'PanelWidthmm':10.10,
        'PanelWindowHeightmm':25.02,
        'PanelHeightmm':25.02+2.9,
    },
    '6300':{
        'PanelLetter':'c',
        'PanelWindowWidthmm':9.70 ,
        'PanelWidthmm':9.70,
        'PanelWindowHeightmm':25.02,
        'PanelHeightmm':25.02+2.9,
    },
    '4278':{
        'PanelLetter':'d',
        'PanelWindowWidthmm':17.50 ,
        'PanelWidthmm':17.50+ 2.68,
        'PanelWindowHeightmm':25.02,
        'PanelHeightmm':25.02+2.9,
    },
    '4861':{
        'PanelLetter':'e',
        'PanelWindowWidthmm':29.52 ,
        'PanelWidthmm':29.52,
        'PanelWindowHeightmm':25.02,
        'PanelHeightmm':25.02+1.94,
    },
    '6563':{
        'PanelLetter':'f',
        'PanelWindowWidthmm':20.0 ,
        'PanelWidthmm':20.0 + 2.68,
        'PanelWindowHeightmm':25.02,
        'PanelHeightmm':25.02+1.94,
    },
}
# %%
wls = predictor.wls
# %%
sum_top = 0
sum_bottom = 0
for widx,wl in enumerate(wls):
    w = str(int(wl*10))
    wdict = Mosaic_ParamDict[w]
    if wdict['PanelLetter'] in ['a','b','c','d']:
        sum_top += wdict['PanelWidthmm']
    else:
        sum_bottom +=  wdict['PanelWidthmm']
# %%
print(sum_top,sum_bottom)
# %%
sum_left = 0
sum_right = 0
for widx,wl in enumerate(wls):
    w = str(int(wl*10))
    wdict = Mosaic_ParamDict[w]
    if wdict['PanelLetter'] in ['a','e']:
        sum_left += wdict['PanelHeightmm']
    elif wdict['PanelLetter'] in ['d','f']:
        sum_right += wdict['PanelHeightmm']

print(sum_left,sum_right)
    
# %%
