#%%
import pickle
import numpy as np
import os
#%%
dirpath = os.path.dirname(__file__)
#%%
hmsA_ParamDict = {
    'hmsVersion':'A',
    'FlCollimator': 400,
    'FlPrimeCamera': 443.401,
    'SlitLengthmm': 55.29,
    'SlitLengthdeg': np.rad2deg(np.arctan(55.29/400)),
    'sigma': 10125.5569,
    'relSlitPositionmm': 21.43,
    'SlitA2FarEdgemm': 63.35,
    'SlitA2CloseEdgemm': 13.83,
    'MosaicWidthmm':52.50,
    'MosaicHeightmm': 55.19,
    'MosaicWindowWidthmm':49.52,
    'MosaicWindowHeightmm':50.05,
    'MosaicFilters': [['4278','6300','5577','7774'], 
                      ['6563','4861']]
}

# Save dictionary to a file
fn = os.path.join(dirpath,'hmsA_Params.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsA_ParamDict, file)
# %%
hmsA_wlParamDict = {
    '5577':{'wl':557.7, 
            'color':'green',
            'SlitNum':2, 
            'DiffractionOrder': 33,
            'PanelLetter':'b',
            'PanelWindowWidthmm':10.10 ,
            'PanelWidthmm':10.10,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '6300':{'wl':630.0, 
            'color':'Red', 
            'SlitNum':2, 
            'DiffractionOrder': 29,
            'PanelLetter':'c',
            'PanelWindowWidthmm':9.70 ,
            'PanelWidthmm':9.70,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '7774':{'wl':777.4, 
            'color':'plum',
            'SlitNum':1, 
            'DiffractionOrder': 24,
            'PanelLetter':'a',
            'PanelWindowWidthmm':12.22 ,
            'PanelWidthmm':12.22,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9
            },
    '4278':{'wl':427.8, 
            'color':'blue', 
            'SlitNum':1, 
            'DiffractionOrder': 43,
            'PanelLetter':'d',
            'PanelWindowWidthmm':17.50 ,
            'PanelWidthmm':17.50+ 2.68,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '6563':{'wl':656.3, 
            'color':'Darkred',
            'SlitNum':3,
            'DiffractionOrder': 28, 
            'PanelLetter':'f',
            'PanelWindowWidthmm':18.83,
            'PanelWidthmm':18.83+ 2.68,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+1.94,
            },
    '4861':{'wl':486.1, 
            'color':'cyan', 
            'SlitNum':4, 
            'DiffractionOrder': 38,
            'PanelLetter':'e',
            'PanelWindowWidthmm':30.69 ,
            'PanelWidthmm':30.69,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+1.94,
            },
}
# Save dictionary to a file
fn = os.path.join(dirpath,'hmsA_wlParams.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsA_wlParamDict, file)

#%%
hmsAEclipse_ParamDict = {
    'hmsVersion':'A - ECLIPSE',
    'FlCollimator': 400,
    'FlPrimeCamera': 429.420,
    'SlitLengthmm': 55.29,
    'SlitLengthdeg': np.rad2deg(np.arctan(55.29/400)),
    'sigma': 10125.5569,
    'relSlitPositionmm': 21.43,
    'SlitA2FarEdgemm': 63.35,
    'SlitA2CloseEdgemm': 13.83,
    'MosaicWidthmm':52.50,
    'MosaicHeightmm': 55.19,
    'MosaicWindowWidthmm':49.52,
    'MosaicWindowHeightmm':50.05,
    'MosaicFilters': [['4278','6300','5577','7774'], 
                      ['6563','4861']]
}

# Save dictionary to a file
fn = os.path.join(dirpath,'hmsAEclipse_Params.pkl')
with open(fn, 'wb') as file:
        pickle.dump(hmsAEclipse_ParamDict,file)

# %%
hmsAEclipse_wlParamDict = {
    '5577':{'wl':557.7, 
            'color':'green',
            'SlitNum':2, 
            'DiffractionOrder': 33,
            'PanelLetter':'b',
            'PanelWindowWidthmm':10.10 ,
            'PanelWidthmm':10.10,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '6300':{'wl':630.0, 
            'color':'Red', 
            'SlitNum':2, 
            'DiffractionOrder': 29,
            'PanelLetter':'c',
            'PanelWindowWidthmm':9.70 ,
            'PanelWidthmm':9.70,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '7774':{'wl':777.4, 
            'color':'plum',
            'SlitNum':1, 
            'DiffractionOrder': 24,
            'PanelLetter':'a',
            'PanelWindowWidthmm':12.22 ,
            'PanelWidthmm':12.22,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9
            },
    '4278':{'wl':427.8, 
            'color':'blue', 
            'SlitNum':1, 
            'DiffractionOrder': 43,
            'PanelLetter':'d',
            'PanelWindowWidthmm':17.50 ,
            'PanelWidthmm':17.50+ 2.68,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '6563':{'wl':656.3, 
            'color':'Darkred',
            'SlitNum':3,
            'DiffractionOrder': 28, 
            'PanelLetter':'f',
            'PanelWindowWidthmm':18.83,
            'PanelWidthmm':18.83+ 2.68,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+1.94,
            },
    '4861':{'wl':486.1, 
            'color':'cyan', 
            'SlitNum':4, 
            'DiffractionOrder': 38,
            'PanelLetter':'e',
            'PanelWindowWidthmm':30.69 ,
            'PanelWidthmm':30.69,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+1.94,
            },
}
# Save dictionary to a file
fn = os.path.join(dirpath,'hmsAEclipse_wlParams.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsAEclipse_wlParamDict, file)
#%%
hmsAOrigin_ParamDict = {
    'hmsVersion':'A-ORIGIN',
    'FlCollimator': 400,
    'FlPrimeCamera': 443.401,
    'SlitLengthmm': 55.29,
    'SlitLengthdeg': np.rad2deg(np.arctan(55.29/400)),
    'sigma': 10125.5569,
    'relSlitPositionmm': ,
    'SlitA2FarEdgemm': 58.35,
    'SlitA2CloseEdgemm': 58.35 -52.50,
    'MosaicWidthmm':52.50,
    'MosaicHeightmm': 55.19,
    'MosaicWindowWidthmm':49.52,
    'MosaicWindowHeightmm':50.05,
    'MosaicFilters': [['4278','6300','5577','7774'], 
                      ['6563','4861']]
}

# Save dictionary to a file
fn = os.path.join(dirpath,'hmsAOrigin_Params.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsAOrigin_ParamDict, file)
# %%
hmsAOrigin_wlParamDict = {
    '5577':{'wl':557.7, 
            'color':'green',
            'SlitNum':2, 
            'DiffractionOrder': 33,
            'PanelLetter':'b',
            'PanelWindowWidthmm':10.10 ,
            'PanelWidthmm':10.10,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '6300':{'wl':630.0, 
            'color':'Red', 
            'SlitNum':2, 
            'DiffractionOrder': 29,
            'PanelLetter':'c',
            'PanelWindowWidthmm':9.70 ,
            'PanelWidthmm':9.70,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '7774':{'wl':777.4, 
            'color':'plum',
            'SlitNum':2, 
            'DiffractionOrder': 24,
            'PanelLetter':'a',
            'PanelWindowWidthmm':12.22 ,
            'PanelWidthmm':12.22,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9
            },
    '4278':{'wl':427.8, 
            'color':'blue', 
            'SlitNum':1, 
            'DiffractionOrder': 43,
            'PanelLetter':'d',
            'PanelWindowWidthmm':17.50 ,
            'PanelWidthmm':17.50+ 2.68,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+2.9,
            },
    '6563':{'wl':656.3, 
            'color':'Darkred',
            'SlitNum':3,
            'DiffractionOrder': 28, 
            'PanelLetter':'f',
            'PanelWindowWidthmm':18.83,
            'PanelWidthmm':18.83+ 2.68,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+1.94,
            },
    '4861':{'wl':486.1, 
            'color':'cyan', 
            'SlitNum':4, 
            'DiffractionOrder': 38,
            'PanelLetter':'e',
            'PanelWindowWidthmm':30.69 ,
            'PanelWidthmm':30.69,
            'PanelWindowHeightmm':25.02,
            'PanelHeightmm':25.02+1.94,
            },
}
# Save dictionary to a file
fn = os.path.join(dirpath,'hmsAOrigin_wlParams.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsAOrigin_wlParamDict, file)
# %%
hmsB_ParamDict = {
    'hmsVersion':'B', #hitmis stayed here for solar eclipse
    'FlCollimator': 400, #focal length of collimator, mm.
    'FlPrimeCamera': 365.891, #focal length of collimator (grating -> mosaic), mm.
    'SlitLengthmm': 55.30, #slight length, mm.
    'SlitLengthdeg': np.rad2deg(np.arctan(55.30/400)), # slit length = slitlen/focallength,Deg.
    'sigma': 10245.208387*100/80.5, #measured grating density
    'relSlitPositionmm': 20.04, #Distance between the two slits, mm.
    'SlitA2FarEdgemm': 71.35, #distance between slit closest to mosaic and the farest edge of the mosaic, mm.
    'SlitA2CloseEdgemm': 15.83, #distance between slit closest to mosaic and the closest edge of the mosaic, mm.
    'MosaicWidthmm':52.52,
    'MosaicHeightmm': 55.10,
    #Blue tap facing away from grating, if we are looking from grating to mosaic
    'MosaicWindowWidthmm':49.96, #The window that the camera images, mm.
    'MosaicWindowHeightmm':50.15, #the window height that the camera images, mm.
    'MosaicFilters': [['4278','7774','6300','5577'],  #top panel from L -> R
                      ['4861','6563']] #bottom panel from L -> R
    }

# Save dictionary to a file
fn = os.path.join(dirpath,'hmsB_Params.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsB_ParamDict, file)

# %%
hmsB_wlParamDict = {
    '5577':{'wl':557.7, 
            'color':'green',
            'SlitNum':2, 
            'DiffractionOrder': 33,
            'PanelLetter':'a',
            'PanelWindowWidthmm':18.18 +1.44,
            'PanelWidthmm':18.18 +1.44,
            'PanelWindowHeightmm':26.69, #measure the window to + or -
            'PanelHeightmm':26.69,
            },
    '6300':{'wl':630.0, 
            'color':'Red', 
            'SlitNum':2, 
            'DiffractionOrder': 29,
            'PanelLetter':'b',
            'PanelWindowWidthmm':7.09+1.44 ,
            'PanelWidthmm':7.09+1.44,
            'PanelWindowHeightmm':26.69, #measure the window to + or -
            'PanelHeightmm':26.69,
            }
            ,
    '7774':{'wl':777.4, 
            'color':'plum',
            'SlitNum':2, 
            'DiffractionOrder': 24,
            'PanelLetter':'c',
            'PanelWindowWidthmm':8.39+1.92,
            'PanelWidthmm':8.39+1.92,
            'PanelWindowHeightmm':26.69, #measure the window to + or -
            'PanelHeightmm':26.69,
            },
    '4278':{'wl':427.8, 
            'color':'blue', 
            'SlitNum':1,
            'DiffractionOrder': 43,
            'PanelLetter':'d',
            'PanelWindowWidthmm':12.89, #measure the window to + or -
            'PanelWidthmm':12.89,
            'PanelWindowHeightmm':26.69, #measure the window to + or -
            'PanelHeightmm':26.69,
            },
    '6563':{'wl':656.3, 
            'color':'Darkred',
            'SlitNum':3,
            'DiffractionOrder': 28,
            'PanelLetter':'e',
            'PanelWindowWidthmm':28.89 + 1.96 , 
            'PanelWidthmm':28.89 + 1.96,
            'PanelWindowHeightmm':26.69, #measure the window to + or -
            'PanelHeightmm':26.69,
            },
    '4861':{'wl':486.1, 
            'color':'cyan', 
            'SlitNum':4, 
            'DiffractionOrder': 38,
            'PanelLetter':'f',
            'PanelWindowWidthmm':20.61 , #measure the window to + or -
            'PanelWidthmm':20.61,
            'PanelWindowHeightmm':26.69, #measure the window to + or -
            'PanelHeightmm':26.69,
            }
}
# Save dictionary to a file
fn = os.path.join(dirpath,'hmsB_wlParams.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsB_wlParamDict, file)
# %%
def load_pickle_file(fn:str):
    with open(fn, 'rb') as file:
        dat = pickle.load(file)
    return dat

checkdict = load_pickle_file('hmsA_Params.pkl')
# %%
checkdict['FlCollimator']

# %%
hmsBOrigin_ParamDict = {
    'hmsVersion':'B-ORIGIN', 
    'FlCollimator': 400, #focal length of collimator, mm.
    'FlPrimeCamera': 365.891, #focal length of collimator (grating -> mosaic), mm.
    'SlitLengthmm': 55.30, #slight length, mm.
    'SlitLengthdeg': np.rad2deg(np.arctan(55.30/400)), # slit length = slitlen/focallength,Deg.
    'sigma': 10245.208387, #measured grating density
#     'sigma': 10131.7122594, #new grating
    'relSlitPositionmm': 20.04, #Distance between the two slits, mm.
    'SlitA2FarEdgemm': 71.35, #distance between slit closest to mosaic and the farest edge of the mosaic, mm.
    'SlitA2CloseEdgemm': 15.83, #distance between slit closest to mosaic and the closest edge of the mosaic, mm.
    'MosaicWidthmm':52.52,
    'MosaicHeightmm': 55.10,
    #Blue tap facing away from grating, if we are looking from grating to mosaic
    'MosaicWindowWidthmm':49.96, #The window that the camera images, mm.
    'MosaicWindowHeightmm':50.15, #the window height that the camera images, mm.
    'MosaicFilters': [['7841', '4278', '6300', '5577'], #top panel from L -> R
                      ['6563','4861']] #,'6544']]  #bottom panel from L -> R
    }

# Save dictionary to a file
fn = os.path.join(dirpath,'hmsBOrigin_Params.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsBOrigin_ParamDict, file)
# %%
hmsBOrigin_wlParamDict = {
    '5577':{'wl':557.7, 
            'color':'green',
            'SlitNum':2, 
            'DiffractionOrder': 33,
            'PanelLetter':'a',
            'PanelWindowWidthmm':14.75,
            'PanelWidthmm':14.75,
            'PanelWindowHeightmm':24.775, #measure the window to + or -
            'PanelHeightmm':24.775+2.64,
            },
    '6300':{'wl':630.0, 
            'color':'Red', 
            'SlitNum':2, 
            'DiffractionOrder': 29,
            'PanelLetter':'b',
            'PanelWindowWidthmm': 6.5,  #1.44 ,
            'PanelWidthmm':6.5,
            'PanelWindowHeightmm':24.775, #measure the window to + or -
            'PanelHeightmm':24.775+2.64,
            },
    '4278':{'wl':427.8, 
            'color':'blue', 
            'SlitNum':1,
            'DiffractionOrder': 43,
            'PanelLetter':'d',
            'PanelWindowWidthmm':6, #measure the window to + or -
            'PanelWidthmm':6,
            'PanelWindowHeightmm':24.775, #measure the window to + or -
            'PanelHeightmm':24.775+2.64,
            },
    '7841':{'wl':784.1, 
            'color':'hotpink', 
            'SlitNum':2,
            'DiffractionOrder': 23,
            'PanelLetter':'d',
            'PanelWindowWidthmm':14.5+8.21, #measure the window to + or -
            'PanelWidthmm':14.5 +8.21 + 2.56,
            'PanelWindowHeightmm':24.775, #measure the window to + or -
            'PanelHeightmm':24.775+2.64,
            },
        
#     '7774':{'wl':777.4, 
#             'color':'plum',
#             'SlitNum':2, 
#             'DiffractionOrder': 23,
#             'PanelLetter':'c',
#             'PanelWindowWidthmm':8.21,
#             'PanelWidthmm': 8.21 + 2.56,
#             'PanelWindowHeightmm':24.775, #measure the window to + or -
#             'PanelHeightmm':24.775+2.64,
#             },
    '4861':{'wl':486.1, 
            'color':'cyan', 
            'SlitNum':4, 
            'DiffractionOrder': 38,
            'PanelLetter':'f',
            'PanelWindowWidthmm':21 , #measure the window to + or -
            'PanelWidthmm':21,
            'PanelWindowHeightmm':25.375, #measure the window to + or -
            'PanelHeightmm':25.375 + 2.31,
            },
    '6563':{'wl':656.3, 
            'color':'Darkred',
            'SlitNum':3,
            'DiffractionOrder': 28,
            'PanelLetter':'e',
            'PanelWindowWidthmm':28.96 , 
            'PanelWidthmm':28.96 + 2.56 ,
            'PanelWindowHeightmm':25.375, #measure the window to + or -
            'PanelHeightmm':25.375 + 2.31,
            },
    
}
# Save dictionary to a file
fn = os.path.join(dirpath,'hmsBOrigin_wlParams.pkl')
with open(fn, 'wb') as file:
    pickle.dump(hmsBOrigin_wlParamDict, file)

