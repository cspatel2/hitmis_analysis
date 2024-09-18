#%%
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
from hitmis_Instrument.img_predictor import HMS_ImagePredictor, load_pickle_file
from hitmis_Instrument.grating import Grating,correct_unit_of_angle
from skimage.transform import warp
from hmspython.Utils._files import *
from hmspython.Utils._Utility import *
from glob import glob
from skimage import exposure
# %%
class MapPixel2Wl:
    def __init__(self, predictor:HMS_ImagePredictor):  
        self.ip = predictor #HMs image predictor
        self.hmsParamDict = self.ip.hmsParamDict
        self.wlParamDict = self.ip.wlParamDict
        self.sigma = self.hmsParamDict['sigma']
    
        print('Calculating Gamma...')
        self.gammagrid = self.get_gamma_grid()
        print('Calculating Beta...')
        self.betagrid = self.get_beta_grid()
        print('Calculating Panel...')
        self.panelgrid = self.get_value_grid('wl')
        print('Calculating Alpha...')
        self.alphagrid = self.get_value_grid('alpha')
        print('Calculating Order of Difraction...')
        self.ordergrid = self.get_value_grid('difractionOrder')
        print('Calculating wavelength map...')
        self.lambdagrid = self.get_lambda_grid()
        print('Done.')

    def calc_lamda_gratingeqn(self,alpha:float,beta:float,gamma:float,order:int) ->float:

        alpha = correct_unit_of_angle(alpha,"rad")
        beta = correct_unit_of_angle(beta,"rad")
        gamma = correct_unit_of_angle(gamma,"rad")

        return (self.sigma/order)*np.sin(gamma)*(np.sin(alpha)+np.sin(beta))

    def get_gamma_grid(self) -> np.array:
        gdeg = 90 + self.ip.mm2deg(self.ip.MosaicWindowHeightmm/2,self.ip.fprime) * np.linspace(-1,1,2*101) #linspace of gammas that are allowed through the mosaic window, one column of gammas
        gdeg_binned = [list(np.linspace(np.min(gdeg),np.max(gdeg),self.ip.pix))] 
        gammadeg_grid = np.array(gdeg_binned*self.ip.pix) #grid
        return np.array(gammadeg_grid.T,dtype = float)
    
    def get_beta_grid(self) -> np.array:

        betafaredge = self.ip.alpha - self.ip.mm2deg(self.hmsParamDict['SlitA2FarEdgemm'],self.ip.fprime)
        betanearedge = betafaredge + self.ip.mm2deg(self.ip.MosaicWindowWidthmm,self.ip.fprime)
        betadeg = [list(np.linspace(betafaredge,betanearedge,self.ip.pix))] #linspace of betas that are allowed through the mosaic, one row of betas
        betadeg_grid = np.array(betadeg*self.ip.pix) #grid
        return np.array(betadeg_grid,dtype = float)
    
    def get_value_grid(self,value:str) -> np.array:
        value = value.lower()
        if value not in ['alpha','wl','difractionorder']: raise ValueError("Value must be 'Alpha', 'wl', 'DifractionOrder. ")
        gammagrid = self.get_gamma_grid() #get grid of gamma values, deg.
        betagrid = self.get_beta_grid() # get grid of beta values, deg.
        
        value_grid = np.zeros((self.ip.pix,self.ip.pix)) #create a grid of zeros.

        for fidx,wllist in enumerate(self.hmsParamDict['MosaicFilters']): #get the list of filter wls
            if fidx == 0: rowidx = np.where(gammagrid[:,0] >= 90)[0] #bottom panel on image
            elif fidx == 1:rowidx = np.where(gammagrid[:,0] <= 90)[0] #top panel on image
            for wlidx,wl in enumerate(wllist):
                wdict = self.wlParamDict[wl] # get wavelength dict
                morder = int(wdict['DiffractionOrder'])

                if wdict['SlitNum'] in [2,4]: alpha = self.ip.alpha_slitA
                elif wdict['SlitNum'] in [1,3]: alpha = self.ip.alpha_slitB

                if wlidx == 0: x = betagrid[0][0] #set starting x value as the left edge of its the first wl in the list
                else: x = x1 #set start x value as the right edge of previous wl
                x1 = x + self.ip.mm2deg(wdict['PanelWindowWidthmm'],self.ip.fprime) # set the right edge of the current wl
                panelmask = (betagrid[0]>=x) & (betagrid[0] <= x1) # find idx within panel
                colidx = np.where(panelmask)[0]
                X, Y = np.meshgrid(rowidx, colidx, indexing='ij')
                
                if value in 'alpha': val = float(alpha)
                elif value in 'wl': val = int(wl)
                elif value in 'difractionorder': val = morder
        
                value_grid[X,Y] = val
        return np.array(value_grid)
        
    def get_lambda_grid(self):
        return list(map(self.calc_lamda_gratingeqn,self.alphagrid,self.betagrid,self.gammagrid,self.ordergrid))

    def get_wlpanel_idx(self,wl,value_grid):
        value_grid = np.array(value_grid)
        rows,cols = np.where((value_grid==wl))
        return rows,cols
        
    def extract_wlpanel(self,wl,value_grid):
        value_grid = np.array(value_grid)
        rows,cols = self.get_wlpanel_idx(wl,self.panelgrid)
        return np.array(value_grid[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1])

    def straighten_img(self, wavelength: float, img:np.array = None, imgpath: str= None, plot: bool = True, plotwlaxis:bool = True, wlrowidx = 50):
        # the wl range of the straightened images should be a row of the image that closer to the center of the mosaic.
        if wavelength in self.ip.hmsParamDict['MosaicFilters'][0]: #bottom panel so beta = 90 is closer to the top of the image
            wlrowidx = np.abs(wlrowidx)
        elif wavelength in self.ip.hmsParamDict['MosaicFilters'][1]:#top panel so beta = 90 is closer to the bottom of the image
            wlrowidx = -np.abs(wlrowidx)
        wl = int(wavelength * 10)
        
        # locate the row that is closes to gamma = 90
        gamma_array = self.extract_wlpanel(wl,self.gammagrid)
        g90idx,_ = find_nearest((gamma_array,90))
        
        # Extract wl panel from raytrace at detector
        wl_array = self.extract_wlpanel(wl, self.lambdagrid)
        rows, cols = np.shape(wl_array)
        wl_min = np.min(wl_array[g90idx])
        wl_max = np.max(wl_array[g90idx])
        # wl_min = np.min(wl_array[wlrowidx])
        # wl_max = np.max(wl_array[wlrowidx])
        # Define target_wls ensuring correct order
        target_wls = np.linspace(wl_max, wl_min, cols)
        print(f"Shape of wl_array: {np.shape(wl_array)}, Shape of target_wls: {np.shape(target_wls)}")
        

        #data can be a 2d array or a filepath to a .fits file
        if isinstance(img,np.ndarray):
            print(f'input img shape = {np.shape(img)}')
            data_panel = self.extract_wlpanel(wl,img)
            cbarlabel = 'Intenisty'
        elif isinstance(imgpath,str):
            if os.path.exists(imgpath):
                # Extract panel from the hms img
                data, _ = open_fits(imgpath)
                # data = exposure.equalize_hist(data)
                data_panel = self.extract_wlpanel(wl, data)
                cbarlabel = 'Normalized Intenisty'
            else: #if path does not exist then make wl
                data_panel = wl_array
                cbarlabel = 'Wavelength'
        else: 
            raise ValueError('Must provide an image array or image file path (.fit or .fits)')
            
        
        # Mapping function for line straightening. It uses target_wls and wl_array from above.
        def mapping_function(xy: np.array):
            xy_transformed = np.empty(np.shape(xy))  # Copy of the coordinate array
            for i in range(len(xy)):
                row, col = int(xy[i, 1]), int(xy[i, 0])
                current_wl = wl_array[row, col]
                # Find column corresponding to the closest wl in target_wls
                target_col = np.argmin(np.abs(target_wls - current_wl))
                xy_transformed[i, 0] = target_col
                xy_transformed[i, 1] = row
            return xy_transformed
        
        # Straighten data using mapping func that is based on simulation
        print(f'Data panel shape: {np.shape(data_panel)}')
        straightened_image = warp(data_panel, mapping_function, preserve_range=True, order=1, mode='constant',cval = np.nan)
        
        # Flip the straightened image horizontally
        straightened_image = np.fliplr(straightened_image)
        target_wls = target_wls[::-1]
        
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot the diffracted spectral lines (curved) on the left
            im0 = axs[0].imshow(data_panel, aspect='auto')
            axs[0].set_title("Diffracted Spectral Lines")
            axs[0].set_xlabel("Pixel Position X")
            axs[0].set_ylabel("Pixel Position Y")
            fig.colorbar(im0, ax=axs[0], label=cbarlabel)

            # Plot the straightened spectral lines on the right
            im1 = axs[1].imshow(straightened_image, aspect='auto')
            axs[1].set_title("Straightened Spectral Lines")
            
            axs[1].set_ylabel("Pixel Position Y")
            fig.colorbar(im1, ax=axs[1], label=cbarlabel)
            centralwlidx = np.argmin(np.abs(target_wls-wavelength))
            axs[1].axvline(x = centralwlidx, color = 'white', linestyle = '--', linewidth = 0.8)
            if plotwlaxis:
                wllabels = np.array([f'{x:.1f}' for x in target_wls])
                l = axs[1].set_xticks(ticks = np.arange(0,len(target_wls),200),labels = wllabels[ np.arange(0,len(target_wls),200)])
                axs[1].set_xlabel("Wavelength [nm]")
            else:
                axs[1].set_xlabel("Pixel Position X")

            plt.tight_layout()
            plt.show()
            
        return straightened_image, target_wls

# %%
# predictor = HMS_ImagePredictor('bo',67.39,50)
# # %%
# mapping = MapPixel2Wl(predictor)

# #%%
# fdir = 'Images/hmsA_img/20240829/*.fits'
# fnames = glob(fdir)
# fnames.sort()
# print(len(fnames))
# # %%
# idx = -300


# # %%
# def bin_and_sum_image(straightened_image, wavelength_bins, row_bin_size=100, col_bin_size=5, plot=True):
#     rows, cols = straightened_image.shape
#     num_row_bins = rows // row_bin_size
#     num_col_bins = cols // col_bin_size

#     # Initialize array to hold binned and summed data
#     binned_data = np.zeros((num_row_bins, num_col_bins))

#     for i in range(num_row_bins):
#         for j in range(num_col_bins):
#             start_row = i * row_bin_size
#             end_row = start_row + row_bin_size
#             start_col = j * col_bin_size
#             end_col = start_col + col_bin_size
#             binned_data[i, j] = np.sum(straightened_image[start_row:end_row, start_col:end_col])

#     # Adjust wavelength bins to match the binned columns
#     binned_wavelength_bins = np.mean(np.reshape(wavelength_bins[:num_col_bins * col_bin_size], (num_col_bins, col_bin_size)), axis=1)

#     if plot:
#         plt.figure(figsize=(12, 6))
#         for i in range(num_row_bins):
#             plt.plot(binned_wavelength_bins, binned_data[i, :], label=f'Bin {i+1}')

#         plt.xlabel("Wavelength")
#         plt.ylabel("Summed Intensity")
#         plt.title("Line Profiles and Read Noise")
#         plt.legend()
#         plt.show()

#     return binned_data, binned_wavelength_bins

# %%
# img,wl_arr =mapping.straighten_img(fnames[idx], 630.0)
# binned_data,wl_arr = bin_and_sum_image(img, wl_arr,col_bin_size=2)
# plt.plot(wl_arr,binned_data.mean(axis = 0),linewidth = 0.7)
# plt.axvline(x = 630.0, linestyle = '--', linewidth = 0.5)
# plt.title(format_time(fnames[idx].split('ccdi_')[-1].strip('.fits')))
# # %%
# plt.figure()

# for i in range(1,5):
#     img,wl =mapping.straighten_img(fnames[idx], 630.0, plot = False)
#     binned_data,wl_arr = bin_and_sum_image(img, wl,col_bin_size=i, plot = False)
    
#     plt.plot(wl_arr,binned_data[7],linewidth = 0.7, label = f'binsize = {i}')
#     plt.xlim(629.5, 630.5)
# plt.axvline(x = 630.0, linestyle = '--', linewidth = 0.5)
# plt.legend(loc = 'best')


    

# # %%
# plt.imshow(mapping.lambdagrid)
# # %%
# img,wl_arr =mapping.straighten_img('',777.4)
# print(np.min(wl_arr), np.max(wl_arr))
# # %%

# %%

# %%
