# %% User-defined variables

# Your path to ctsm_py directory (i.e., where utils.py lives)
sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")

# Directory where input file(s) can be found
indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/"

# Either the name of a file within $indir, or a pattern that will return a list of files.
pattern = "*h1.*-01-01-00000.nc"


# %% Setup

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import utils


# %% Import dataset

# import importlib
# importlib.reload(utils)

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
filelist = glob.glob(indir + pattern)

# Import
# this_ds = utils.import_ds(filelist, myVars=myVars)
this_ds = utils.import_ds(filelist, myVars=myVars, myVegtypes=utils.define_mgdcrop_list())


# %% Read one variable from dataset. (Do nothing with it.)

# import importlib
# importlib.reload(utils)

# Which variable?
thisVar = "CPHASE"

thisvar_da = utils.get_thisVar_da(thisVar, this_ds)
thisvar_da = utils.trim_da_to_mgd_crop(thisvar_da, this_ds.patches1d_itype_veg_str)


# %% Grid and make map, more efficiently, as function

# import importlib
# importlib.reload(utils)

# Grid
# tmp_vyx = utils.grid_one_variable(this_ds, "CPHASE", time=181)
tmp_vyx = utils.grid_one_variable(this_ds, "CPHASE", time="2000-07-01")

# Make map
tmp_yx = tmp_vyx.sel(ivt_str="temperate_corn")
if tmp_yx.shape[0] == 1:
    tmp_yx = tmp_yx.squeeze()
else:
    raise ValueError("You must select one time step to plot")
tmp_yx = utils.cyclic_dataarray(tmp_yx)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolor(tmp_yx.lon.values, tmp_yx.lat.values, tmp_yx, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()

# # Get dates in a format that matplotlib can use
# with warnings.catch_warnings():
#     # Ignore this warning in this with-block
#     warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar, 'noleap', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.")
#     datetime_vals = this_ds.indexes["time"].to_datetimeindex()