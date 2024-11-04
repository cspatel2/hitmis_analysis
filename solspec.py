# %%
import numpy as np
import xarray as xr
from skmpython import vac2air
# %%
data = np.loadtxt('solarspectra.txt').T
# %%
ds = xr.Dataset(
    data_vars = {
        'irradiance': xr.Variable('wavelength', data[1], {'unit': 'W/m^2/nm', 'description': 'Solar irradiance'}),
    },
    coords = {
        'wavelength': xr.Variable('wavelength', vac2air(data[0]), {'unit': 'nm', 'description': 'Air Wavelength'}),
    },
    attrs = {
        'source': 'Chance & Kurucz (2005)',
        'doi': 'doi:10.1016/j.jqsrt.2010.01.036',
        'url': 'http://kurucz.harvard.edu/sun/irradiance2005/irradthu.dat',
    }
)
ds.to_netcdf('solar_spectra_air.nc', encoding = {
    'irradiance': {'zlib': True},
    'wavelength': {'zlib': True}
})
# %%
xr.Variable('wavelength', data[1], {'unit': 'W/m^2/nm', 'description': 'Solar irradiance'})
# %%
