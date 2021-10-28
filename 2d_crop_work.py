
# %% Setup

import numpy as np
import xarray as xr
from xarray.backends.api import load_dataset
from ctsm_py import utils
import matplotlib.pyplot as plt
import warnings
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")
from utils import import_ds_from_filelist, get_thisVar_da, trim_to_mgd_crop, grid_one_timestep, pftlist, cyclic_dataarray



# Import dataset

# Define list of variables to import
myVars = ["CPHASE", \
          "GDDHARV", 
          "GDDPLANT", 
          "GPP", 
          "GRAINC_TO_FOOD", 
          "NPP", 
          "TLAI", 
          "TOTVEGC"]

# Get list of all files in $indir matching $pattern
indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/"
pattern = "*h1.*-01-01-00000.nc"
filelist = glob.glob(indir + pattern)

# Import
this_ds, vegtypes = import_ds_from_filelist(filelist, pftlist, myVars)

# Get dates in a format that matplotlib can use
with warnings.catch_warnings():
    # Ignore this warning in this with-block
    warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar, 'noleap', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.")
    datetime_vals = this_ds.indexes["time"].to_datetimeindex()

# Get PFT list, integers (use only first timestep)



# %% Read one variable from dataset. (Do nothing with it.)

# Which variable?
thisVar = "CPHASE"

thisvar_da = get_thisVar_da(thisVar, this_ds, vegtypes["str"])
thisvar_da = trim_to_mgd_crop(thisvar_da)
thisvar_da


# %% Grid and make map, more efficiently, as function

# Grid
tmp_pyx = grid_one_timestep(this_ds, "pfts1d_itype_veg", 0, vegtypes)

# Make map
tmp_yx = tmp_pyx.isel(pft=0)
tmp_yx = cyclic_dataarray(tmp_yx)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolor(tmp_yx.lon.values, tmp_yx.lat.values, tmp_yx, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()
