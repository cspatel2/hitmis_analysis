#%% 
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
import matplotlib.patches as patches
from tqdm import tqdm
from hmspython.Utils._Utility import *

# %%
def single_defraction(alpha:float,gamma:float,m_order:int, sigma:float, wl:float)-> float:
    if isinstance(m_order, Iterable):
        m_iter = list(m_order)
    else:
        m_iter = [m_order]

    if isinstance(gamma, Iterable):
        beta_arr = []
        for m in m_iter:
            betas = [single_defraction(alpha,g_,m,sigma,wl) for g_ in gamma] 
            beta_arr.append(betas)
        return np.asanyarray(beta_arr)

            
    alpha_rad = correct_unit_of_angle(alpha,'rad')
    gamma_rad = correct_unit_of_angle(gamma,'rad')  # noqa: F405

    beta_rad = np.arcsin(-np.sin(alpha_rad) + ((m_order*wl)/(sigma*np.sin(gamma_rad))) )

    return np.rad2deg(beta_rad)

def Grating(alpha: float, gamma: float, m_order: int, sigma: float, wl: float) -> float:
    if isinstance(alpha, Iterable):
        alpha_iter = list(alpha)
    else:
        alpha_iter = [alpha]

    if isinstance(gamma, Iterable):
        gamma_iter = list(gamma)
    else:
        gamma_iter = [gamma]
    if isinstance(wl, Iterable):
        wl_iter = list(wl)
    else:
        wl_iter = [wl]
    

    results = []
    
    for wl in wl_iter:
        res_per_wl = []
        for alpha in alpha_iter:
            beta = single_defraction(alpha,gamma_iter,m_order,sigma,wl)
            res_per_wl.append(beta)
        results.append(res_per_wl)
    
    # shape(result) = (len(wl), len(alpha), 2*m_order + 1, len(gamma))
    return np.asanyarray(results) 

def Plot_GratingDeffraction(betas:Iterable,wl:Iterable,alpha:Iterable,m_order:int,gamma:Iterable, save_fig:bool = False,multi_order:bool = True, mosaic = True):
    shape = np.shape(betas)
    if isinstance(alpha, Iterable):
        alpha_iter = list(alpha)
    else:
        alpha_iter = [alpha]

    if isinstance(gamma, Iterable):
        gamma_iter = list(gamma)
    else:
        gamma_iter = [gamma]
    if isinstance(m_order, Iterable):
        m_iter = list(m_order)
    else:
        m_iter = [m_order]        

    if isinstance(wl, Iterable):
        wl_iter = list(wl)
    else:
        wl_iter = [wl]


    #define colors for each wavelength. they correspond to the closest color of wl
    colors_dict = {'6300':'red', '5577':'green', '4861':'cyan', '6563': 'darkred', '7774':'plum', '4278':'blue'}
    
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set_title(f'Defracted Spectral lines at Alpha = {alpha_iter[0]} deg')
    ax.set_xlabel('Beta [Deg]')
    ax.set_ylabel('Gamma [Deg]')
    ax.set_xlim(90.0, -90.0)
    ax.set_ylim(gamma[0] - 5, gamma[-1]+5)
    ax.grid(True)

    #plot slit
    for a in alpha_iter:
        plt.plot([a]*2, [np.min(gamma)-1, np.max(gamma)+1], color = 'black')
        ax.text(float(a),np.min(gamma)-1.5, f'Slit', color = 'black')

    def mm2betadeg(len_mm):
        fl = 400 #mm
        return np.rad2deg(len_mm/fl)
    
    if mosaic: # this assumes there is only one slit in hitmis
        #postion of green(557.7) panel
        rect1 = patches.Rectangle((a-mm2betadeg(69.16), 90), mm2betadeg(18.18), 5, linewidth=1, edgecolor=colors_dict['5577'], facecolor='none')
        ax.add_patch(rect1)
        #postion of red(630.0) panel
        rect2 = patches.Rectangle((a-mm2betadeg(49.16), 90), mm2betadeg(7.09), 5, linewidth=1, edgecolor=colors_dict['6300'], facecolor='none')
        ax.add_patch(rect2)
        #postion of 777.4 nm panel
        rect3 = patches.Rectangle((a-mm2betadeg(40.15), 90), mm2betadeg(8.39), 5, linewidth=1, edgecolor=colors_dict['7774'], facecolor='none')
        ax.add_patch(rect3)
        #postion of Blue(427.8) panel
        rect4 = patches.Rectangle((a-mm2betadeg(30.32), 90), mm2betadeg(12.89), 5, linewidth=1, edgecolor=colors_dict['4278'], facecolor='none')
        ax.add_patch(rect4)

        #postion of H-alpha (656.3) panel
        rect5 = patches.Rectangle((a-mm2betadeg(69.16), 90), mm2betadeg(20.16), -5, linewidth=1, edgecolor=colors_dict['6563'], facecolor='none')
        ax.add_patch(rect5)
        #postion of H-beta(486.1) panel
        rect6 = patches.Rectangle((a-mm2betadeg(46.59), 90), mm2betadeg(28.89), -5, linewidth=1, edgecolor=colors_dict['4861'], facecolor='none')
        ax.add_patch(rect6)

    

    for widx,wl in enumerate(wl_iter):
        w = str(int(wl*10)) # nm -> A
        if w not in colors_dict.keys():
            color = 'Black'
        else:
            color = colors_dict[w]
        for aidx, alpha in enumerate(alpha_iter):
            for midx,m in enumerate(m_iter):
                beta = betas[widx][aidx][midx]
                plt.scatter(beta,gamma,s=0.8,color = color)
                ax.text(beta[-1], gamma[-1]+0.5, f'({aidx},{m}, {wl})', ha ='left',fontsize = 8, color = color,rotation=90, rotation_mode='anchor', 
         transform_rotates_text=True)
    
    # plt.show()
    if save_fig:
        plt.savefig(f"grating_result_{alpha_iter[0]}.png")
    
#%%

if __name__ == '__main__':
    
    slit_length = 55.30 #slight length, mm.
    fl_collimator = 400 #focal length of collimator, mm.
    slit_length_gamma = np.rad2deg(np.arctan(slit_length/(fl_collimator)))  # slit length in gamma, Deg.
    # grating_density = 98.76 #lines/mm.
    # sigma = 1e6/grating_density #distace b/w lines, nm.
    # Line density of grating
    sigma = 10248.208387 #measured grating density

    num_orders = 75  # Difraction order.
    orders = np.arange(-num_orders,num_orders+1,1,dtype = int) # all int orders between -num_orders and num_orders.
    samples = 50 # Samples per spectral line.

    # wls = [486.1, 557.7, 630.0, 656.3, 777.4]  # nm
    wls = [630.0,557.7]
    alphas = [83.5]
    gammas = 90 + ((slit_length_gamma/2) * np.linspace(-1, 1, 2 * samples + 1))
     #range of gamma allowed through the slit, Deg. gamma meaured perp to grating. 

    
    #Calculate results of the grating despersion
    result = Grating(alphas, gammas, orders, sigma, wls)

    #Should print (len(wl), len(alpha), 2*m_order + 1, len(gamma))
    print(result.shape)  

    #plot the results 
    Plot_GratingDeffraction(result,wls,alphas,orders,gammas,True)


# %%
def hms2_GratingPredictor(alpha:float = 83.5, wls:Iterable=[486.1, 427.8, 557.7, 630.0, 656.3, 777.4],num_orders:int = 75, samples:int = 50, Mosaic = True):

    ## known Hitmis 2 parameters
    slit_length = 55.30 #slight length, mm.
    fl_collimator = 400 #focal length of collimator, mm.
    slit_length_gamma = np.rad2deg(np.arctan(slit_length/(fl_collimator)))  # slit length in gamma, Deg.
    sigma = 10248.208387 #measured grating density 
    alpha_slitA = alpha
    alpha_slitB = alpha_slitA + mm2betadeg(21.07)
    alphas = [alpha_slitA,alpha_slitB]
    gamma = 90 + ((slit_length_gamma/2) * np.linspace(-1, 1, 2 * samples + 1))
    
    orders = np.arange(-num_orders,num_orders+1,1,dtype = int) # all int orders between -num_orders and num_orders.

    ## get beta values using grating equation
    betas = Grating(alphas, gamma, orders, sigma, wls)
    fig, ax = plt.subplots(figsize=(20, 4),dpi = 300)
    ax.set_xlim(90.0, -90.0)
    
    ax.set_ylim(gamma[0] - 5, gamma[-1]+5)
    ax.grid(True)

    ax.set_title(f'Defracted Spectral lines at Alpha = {alphas[0]} deg')
    ax.set_xlabel('Beta [Deg]')
    ax.set_ylabel('Gamma [Deg]')


    colors_dict = {'5577':'green','6300':'red', '7774':'plum', '4278':'blue', '6563': 'darkred', '4861':'cyan'}
    alpha_dict = {'5577':0,'6300':0, '7774':0, '4278':1, '6563': 1, '4861':0} #0 = slitA, #1 = slitB
    slit_dict = {'5577':2,'6300':2, '7774':2, '4278':1, '6563': 3, '4861':4} # 2,3=slitA, #1,4 = slitB
    gamma_mask_dict =  {'5577':(gamma>=90),'6300':(gamma>=90), '7774':(gamma>=90), '4278':(gamma>=90), '6563': (gamma<=90), '4861':(gamma<=90)}

    m_iter = orders
    wl_iter = wls
    alpha_iter = alphas

    for widx,wl in enumerate(wl_iter):
        w = str(int(wl*10)) # nm -> A
        if w not in colors_dict.keys():
            color = 'Black'
        else:
            color = colors_dict[w]
        aidx = alpha_dict[w]
        gmask = gamma_mask_dict[w]
        gamma_plot = gamma[gmask]
        for midx,m in enumerate(m_iter):
            beta = betas[widx][aidx][midx]
            beta_plot = beta[gmask]
            plt.scatter(beta_plot,gamma_plot,s=0.8,color = color)

            if gmask[0]: #bottom panel, text at gamma min
                ax.text(beta_plot[0], np.min(gamma_plot)-0.5, f'({slit_dict[w]},{m})', ha ='left',fontsize = 8, color = color,rotation=270, rotation_mode='anchor',transform_rotates_text=True)
            else: #top panel, text at gamma max
                ax.text(beta_plot[-1], np.max(gamma_plot)+0.5, f'({slit_dict[w]},{m})', ha ='left',fontsize = 8, color = color,rotation=90, rotation_mode='anchor',transform_rotates_text=True)
        
    #plot slit
    slit_letter = ['A', 'B']
    for aidx, a in enumerate(alpha_iter):
        plt.plot([a]*2, [np.min(gamma), np.max(gamma)], color = 'black')
        ax.text(a, np.max(gamma)+1.5, f'Slit {slit_letter[aidx]}',fontsize = 8, color = 'black',rotation=90, rotation_mode='anchor',transform_rotates_text=True)

    if Mosaic: # this assumes there is only one slit in hitmis
        
        #postion of green(557.7) panel
        height = mm2betadeg(26.01)
        a = alpha_iter[0]
        rect1 = patches.Rectangle((a-mm2betadeg(69.16), 90), mm2betadeg(18.18), height, linewidth=1, edgecolor=colors_dict['5577'], facecolor='none')
        ax.add_patch(rect1)
        #postion of red(630.0) panel
        rect2 = patches.Rectangle((a-mm2betadeg(49.16), 90), mm2betadeg(7.09), height, linewidth=1, edgecolor=colors_dict['6300'], facecolor='none')
        ax.add_patch(rect2)
        #postion of 777.4 nm panel
        rect3 = patches.Rectangle((a-mm2betadeg(40.15), 90), mm2betadeg(8.39), height, linewidth=1, edgecolor=colors_dict['7774'], facecolor='none')
        ax.add_patch(rect3)
        #postion of Blue(427.8) panel
        rect4 = patches.Rectangle((a-mm2betadeg(30.32), 90), mm2betadeg(12.89), height, linewidth=1, edgecolor=colors_dict['4278'], facecolor='none')
        ax.add_patch(rect4)

        #postion of H-alpha (656.3) panel
        rect5 = patches.Rectangle((a-mm2betadeg(69.16), 90), mm2betadeg(28.89), -height, linewidth=1, edgecolor=colors_dict['6563'], facecolor='none')
        ax.add_patch(rect5)
        #postion of H-beta(486.1) panel
        rect6 = patches.Rectangle((a-mm2betadeg(38.31), 90), mm2betadeg(28.89), -height, linewidth=1, edgecolor=colors_dict['4861'], facecolor='none')
        ax.add_patch(rect6)
    # plt.show()
    plt.savefig(f'grating_{alpha}.png')
    del fig

