#%%
import numpy as np
import matplotlib.pyplot as plt
from hmspython.Diffraction._ImgPredictor import HMS_ImagePredictor
from hmspython.Utils._files import *
from hmspython.Utils._Utility import *
from glob import glob
from skimage import transform
import os
# %%
class MapPixel2Wl:
    """This Class maps the each pixel of the detector to the corresponding wavelegth that is indicent on it though the hms instrument.
    """    
    def __init__(self, predictor:HMS_ImagePredictor) -> None:  
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
        print('Calculating Order of diffraction...')
        self.ordergrid = self.get_value_grid('diffractionOrder')
        print('Calculating wavelength map...')
        self.lambdagrid = self.get_lambda_grid()
        print('Done.')

    def calc_lamda_gratingeqn(self,alpha:float,beta:float,gamma:float,order:int) ->float:
        """Calulates λ using the grating equation \n λ = σ(sinγ)/m * (sinα + sinβ) 
        Args:
            alpha (float): angle of incidence perpendicluar to groves, α [Degrees (0-360) or Radians(0-2π)].
            beta (float): angle of diffraction, β [Degrees (0-360) or Radians(0-2π)].
            gamma (float): angle of incidence parallel to groves,γ [Degrees (0-360) or Radians(0-2π)]. 
            order (int): diffraction order, m.

        Returns:
            float: Wavelength, λ.
        """        
        alpha = correct_unit_of_angle(alpha,"rad")  # noqa: F405
        beta = correct_unit_of_angle(beta,"rad")
        gamma = correct_unit_of_angle(gamma,"rad")

        return (self.sigma/order)*np.sin(gamma)*(np.sin(alpha)+np.sin(beta))

    def calc_resolution_gratingeqn(self,alpha:float,beta:float,gamma:float,slitwidth:float,wl:float) -> float:
        """ Calulates spectral resolution of hms given the slitwidth at wavelength λ using the grating equation: \n Δλ = λ*B*cos(β)*[sin(γ)]^(-1) * [sin(α) + sin(β)]^(-1) 

        Args:
            alpha (float): angle of incidence perpendicluar to groves, α [Degrees (0-360) or Radians(0-2π)].
            beta (float): angle of diffraction, β [Degrees (0-360) or Radians(0-2π)].
            gamma (float): angle of incidence parallel to groves,γ [Degrees (0-360) or Radians(0-2π)]. 
            slitwidth (float): Width of the entrance silt, [microns](1e-6).
            wl (float):  Wavelength, λ [nm]
            
        Returns:
            float: spectral resolution [nm] 
        """    
        alpha = correct_unit_of_angle(alpha,'rad')
        beta = correct_unit_of_angle(beta,'rad')
        gamma = correct_unit_of_angle(gamma,'rad')
        wl = wl*1e-9 #nm -> m
        slitwidth = slitwidth *1e-6 #microns -> m
        return( wl*slitwidth*np.cos(beta)/np.sin(gamma) * (np.sin(alpha) + np.sin(beta))**(-1))*1e9
    def get_gamma_grid(self) -> np.array:
        """ calculates angle of incidence parallel to grooves for all pixel postions.

        Returns:
            np.array: gamma_grid [degrees], shape = (totalpix X totalpix)
        """        
        gdeg = 90 + self.ip.mm2deg(self.ip.MosaicWindowHeightmm/2,self.ip.fprime) * np.linspace(-1,1,2*101) #linspace of gammas that are allowed through the mosaic window, one column of gammas
        gdeg_binned = [list(np.linspace(np.min(gdeg),np.max(gdeg),self.ip.pix))] 
        gammadeg_grid = np.array(gdeg_binned*self.ip.pix) #grid
        return np.array(gammadeg_grid.T,dtype = float)
    
    def get_beta_grid(self) -> np.array:
        """calculates angle of diffraction for all pixel postions.

        Returns:
            np.array:  beta_grid [degrees], shape = (totalpix X totalpix)
        """        
        betafaredge = self.ip.alpha - self.ip.mm2deg(self.hmsParamDict['SlitA2FarEdgemm'],self.ip.fprime)
        betanearedge = betafaredge + self.ip.mm2deg(self.ip.MosaicWindowWidthmm,self.ip.fprime)
        betadeg = [list(np.linspace(betafaredge,betanearedge,self.ip.pix))] #linspace of betas that are allowed through the mosaic, one row of betas
        betadeg_grid = np.array(betadeg*self.ip.pix) #grid
        return np.array(betadeg_grid,dtype = float)
    
    def get_value_grid(self,value:str) -> np.array:
        """calculates value for requested variable for all pixel postions. variables can be one of the following: 'alpha','wl','diffractionorder'

        Args:
            value (str): Requestion variable can be: \n 'alpha' - incidence angle perpendicular to grooves (degrees). \n 'wl' - the wl that the panel belongs to (Angstroms). \n 'diffractionorder' - order of diffraction m.

        Raises:
            ValueError: Value must be 'Alpha', 'wl', 'diffractionOrder'. 

        Returns:
            np.array: grid of values of shape = (totalpix X totalpix)
        """        
        value = value.lower()
        if value not in ['alpha','wl','diffractionorder']: raise ValueError("Value must be 'Alpha', 'wl', 'diffractionOrder. ")
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
                elif value in 'diffractionorder': val = morder
        
                value_grid[X,Y] = val
        return np.array(value_grid)
        
    def get_lambda_grid(self) -> list[float]:
        """ Calculate diffracted wavelength for all pixel postions.

        Returns:
            list[float]: wavelegth array of shape totalpix X totalpix
        """        
        return list(map(self.calc_lamda_gratingeqn,self.alphagrid,self.betagrid,self.gammagrid,self.ordergrid))
    
    def get_resolution_grid(self, slitwidth:float) -> list[float]:
        """Calculate spectral resolution for all pixel postions.

        Args:
            slitwidth (float):  Width of the entrance silt, [microns](1e-6).

        Returns:
            list[float]: spectral resolution array of shape totalpix X totalpix
        """  
        b = np.empty_like(self.alphagrid)    
        b.fill(slitwidth)  
        return list(map(self.calc_resolution_gratingeqn,self.alphagrid,self.betagrid,self.gammagrid,b,self.lambdagrid))

    def get_wlpanel_idx(self,wl:int,value_grid:np.ndarray) -> tuple[np.array,np.array]:
        """ find idices corresponding to the wavelength (int, A). Used to find the pixels that correspond to single ROI or panel.

        Args:
            wl (int): Wavelength, Angstroms.
            value_grid (np.ndarray): grid of values of shape = (totalpix X totalpix). Should be used with self.panelgrid.

        Returns:
            tuple[np.array,np.array]: rows, columns
        """        
        value_grid = np.array(value_grid)
        rows,cols = np.where((value_grid==wl))
        return rows,cols
        
    def extract_wlpanel(self,wl:int,value_grid:np.ndarray) -> np.ndarray:
        """ extracts the value_grid that corresponds to the ROI of the wavelength.

        Args:
            wl (int): Wavelength, Angstroms.
            value_grid (np.ndarray): grid of values of shape = (totalpix X totalpix).

        Returns:
            np.ndarray: ROI_value_grid of shape (n,m)
        """        
        
        value_grid = np.array(value_grid)
        rows,cols = self.get_wlpanel_idx(wl,self.panelgrid)
        return np.array(value_grid[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1])

    def straighten_img(self,wavelength:float,img:np.ndarray|str,rotate_deg:float = 0, plot: bool =  True, plotwlaxis:bool= True) -> tuple[np.ndarray, np.ndarray, np.array]:
        """ straighten the img by panel.

        Args:
            wavelength (float): wavelength of ROI, nm.
            img (np.ndarray | str): HMS img. it can be a 2D array or a str path to .fits file. If path does not exist, it will straighten the modeled spectra.
            rotate_deg (float, optional): rotate the orginal HMS image counter clockwise from the center, degrees. Defaults to 0.
            plot (bool, optional): if true, plot the input image on the right, and straighted image on the left. Defaults to True.
            plotwlaxis (bool, optional): if true, it plots wavelength on the x-axis of the straighted img. Defaults to True.

        Raises:
            ValueError: img must be a 2D array or a str.

        Returns:
            tuple[np.ndarray, np.ndarray, np.array]: straighted img, input image, wavelength-array of straighted img. 
        """        
        wl = int(wavelength*10)
        ## extract panel of calcluated lambdas (Modeled)
        wl_grid = self.extract_wlpanel(wl,self.lambdagrid)
        
        ##  extract panel from curved img 
        if isinstance(img,np.ndarray):
            img = transform.rotate(img,rotate_deg,cval = np.nan, preserve_range=True)
            curved_img_grid = self.extract_wlpanel(wl,img)
            cbarlabel = 'Intensity [ADU]'
        elif isinstance(img,str):
            if os.path.exists(img):# Extract panel from the hms img
                img, _ = open_fits(img)
                img = transform.rotate(img,rotate_deg,cval = np.nan, preserve_range=True)
                curved_img_grid = self.extract_wlpanel(wl, img)
                cbarlabel = 'Intensity [ADU]'
            else: # if file does not exists, straighten model wl_grid
                curved_img_grid = wl_grid
                cbarlabel = 'Wavelength [nm]'
        else: raise ValueError('img must be a 2D array or str img path.')
                
            
        # Chose the target wl array closest to beta = 90 deg to straighted img against.
        rows,cols  = np.shape(wl_grid)
        gamma_grid = self.extract_wlpanel(wl,self.gammagrid)
        gidx,_ = find_nearest(gamma_grid[:,50], self.ip.mgammadeg)
        target_arr = wl_grid[gidx,:]
        target_wls = target_arr
        # target_wls = np.linspace(np.nanmin(target_arr), np.nanmax(target_arr), cols)
        
        #straighten img
        straight_img = []
        for k in range(rows):
            current_row = wl_grid[k]
            corrected_row = np.array([np.nan]*np.shape(current_row)[0])
            for idx,val in enumerate(current_row):
                newidx,_ = find_nearest(target_wls,val)
                # newidx = np.nanargmin(np.abs(target_wls-val))
                corrected_row[newidx] = curved_img_grid[k,idx]
            straight_img.append(corrected_row)
            
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot the diffracted spectral lines (curved) on the left
            im0 = axs[0].imshow(curved_img_grid, aspect='auto')
            axs[0].set_title("Diffracted Spectral Lines")
            axs[0].set_xlabel("Pixel Position X")
            axs[0].set_ylabel("Pixel Position Y")
            fig.colorbar(im0, ax=axs[0], label=cbarlabel)
            
            # Plot the straightened spectral lines on the right
            im1 = axs[1].imshow(straight_img, aspect='auto')
            axs[1].set_title("Straightened Spectral Lines")
            
            axs[1].set_ylabel("Pixel Position Y")
            fig.colorbar(im1, ax=axs[1], label=cbarlabel)
            centralwlidx = np.argmin(np.abs(target_wls-wavelength))
            axs[1].axvline(x = centralwlidx, color = 'white',
            linestyle = '--', linewidth = 0.5)
            if plotwlaxis:
                wllabels = [f'{x:.1f}' for x in target_wls]
                stepsize = 200
                l = axs[1].set_xticks(ticks = np.arange(0,len(target_wls),stepsize),labels = wllabels[::stepsize])
                axs[1].set_xlabel("Wavelength [nm]")
            else:
                axs[1].set_xlabel("Pixel Position X")

            plt.tight_layout()
            plt.show()
        
        return np.array(straight_img), np.array(curved_img_grid), np.array(target_wls)

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
