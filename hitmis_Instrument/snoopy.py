#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#%%
# Given data

x = [656.28, 486.1, 635, 543.5 ,587.5, 667.8, 687.7, 504.7, 501.5, 0]
y = [93003, 73829, 90355, 80118, 85309, 94176, 98492, 75924, 75563, 17117]
# x = [656.28, 486.1, 635, 543.5 ,587.5, 667.8, 504.7, 501.5, 0]
# y = [93003, 73829, 90355, 80118, 85309, 94176, 75924, 75563, 17117]
# p0 = [1/0.00861013, 17225]


# Define the linear model
def linear_model(x, m, b):
    x = np.array(x)
    return m * x + b

# Perform the curve fit

popt, pcov = curve_fit(linear_model, x, y, p0)
m_opt, b_opt = popt

# Calculate errors on the fitted parameters (standard deviations)
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr

# Calculate fitted y values
y_fit = linear_model(x, m_opt, b_opt)

# Calculate the R^2 value
ss_res = np.sum((y - y_fit) ** 2) #sum of sqaure residuals (SSR)
ss_tot = np.sum((y - np.mean(y)) ** 2) #sum of squares (SST)
r_squared = 1 - (ss_res / ss_tot)

print(f"Optimal m: {m_opt} ± {m_err}")
print(f"Optimal b: {b_opt} ± {b_err}")
print(f"R^2 value: {r_squared}")

# %%
tribar_string = "≡"
plt.figure()
wl_range = np.arange(-10,700,20)
plt.plot(wl_range,linear_model(wl_range,m_opt,b_opt), label = 'Fit')
plt.scatter(x,y, marker = 'x', color = 'red', label = 'Measurments')
plt.xlim(np.min(wl_range),np.max(wl_range))
plt.xlabel("Wavelength [nm]")
plt.ylabel("Monochromator Step")
plt.legend(loc = "best")
plt.text(400, min(y)+8000, f'R² = 1 - {ss_res / ss_tot:.5e} \n0 nm {tribar_string} {b_opt:.2f} ± {b_err:.2f} \n1 step {tribar_string} {1/m_opt:.8f} ± {1/m_err:.2f} nm', fontsize=9, verticalalignment='top')


# %%


# %%
