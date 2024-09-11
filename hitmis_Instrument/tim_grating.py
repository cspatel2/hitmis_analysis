#%%
import numpy as np
import matplotlib.pyplot as plt

def beta_degree_fn(d, n, wavelength, alpha_degree, gamma_degree):
    return np.degrees(np.arcsin(n * wavelength / (d * np.sin(np.radians(gamma_degree))) - np.sin(np.radians(alpha_degree))))

def grating():
    # Wavelengths
    # lambdas = np.array([427.8, 486.1, 557.7, 630.0, 656.3, 777.4])
    lambdas = np.array([557.7, 630.0])

    
    grating_angle = 4.0

    # Line density of grating
    # d = 12820.5
    grating_density = 98.76 #lines/mm.
    d = 1e6/grating_density #distace b/w lines, nm.
    # Line density of grating

    # Slit location in alpha
    # alphas = np.array([62.5, 60.14]) + grating_angle
    alphas = np.array([80])
    slit_length_gamma = 10.0
    num_orders = 75
    num_lambdas = len(lambdas)
    num_slits = len(alphas)
    num = 50
    num_points = 2 * num + 1
    gamma_array = 90.0 + (slit_length_gamma / 2.0) * (np.arange(num_points) - num) / num
    alpha_arrays = np.tile(alphas[:, np.newaxis], (1, num_points))
    beta_arrays = np.zeros((num_slits, num_lambdas, num_orders, num_points))

    # Plot colors
    # colors = ['blue', 'cornflowerblue', 'darkgreen', 'limegreen', 'orange', 'red']/
    colors = ['green', 'red']


    for s in range(num_slits):
        for j, wavelength in enumerate(lambdas):
            for n in range(num_orders):
                beta_arrays[s, j, n, :] = beta_degree_fn(d, n, wavelength, alphas[s], gamma_array)

    # Plot settings
    fig, ax = plt.subplots(figsize=(20, 4))

    ax.set_xlim(90.0, -90.0)
    ax.set_ylim(90.0 - 7.0, 90.0 + 7.0)
    ax.set_title('Lines')
    ax.set_xlabel('beta')
    ax.set_ylabel('gamma')

    for s in range(num_slits):
        ax.plot(alpha_arrays[s, :], gamma_array, color='red', linewidth=0.8)
        ax.axhline(y=alphas[s], color='gray', linewidth=0.8)
        ax.axhline(y=-alphas[s], color='gray', linewidth=0.8)
    
    


    for s in range(num_slits):
        for i, wavelength in enumerate(lambdas):
            color = colors[i]
            for n in range(num_orders):
                beta_array = beta_arrays[s, i, n, :]
                ax.plot(beta_array, gamma_array, color=color, linewidth=0.8)
                ax.text(beta_array[-1], gamma_array[-1] + 0.25, f"{n:02}", ha='center', va='center', fontsize=8, color=colors[i])
                ax.text(beta_array[0], gamma_array[0] - 0.25, f"{wavelength:.1f}", ha='center', va='center', fontsize=8, color=colors[i], rotation=180)

    plt.show()

grating()


# %%
