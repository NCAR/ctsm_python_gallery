
# %% Setup

import numpy as np
import xarray as xr
from xarray.backends.api import load_dataset
from ctsm_py import utils
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")
from utils import cyclic_dataarray

pftname =   ["not_vegetated",
             "needleleaf_evergreen_temperate_tree",
             "needleleaf_evergreen_boreal_tree",
             "needleleaf_deciduous_boreal_tree",
             "broadleaf_evergreen_tropical_tree",
             "broadleaf_evergreen_temperate_tree",
             "broadleaf_deciduous_tropical_tree",
             "broadleaf_deciduous_temperate_tree",
             "broadleaf_deciduous_boreal_tree",
             "broadleaf_evergreen_shrub",
             "broadleaf_deciduous_temperate_shrub",
             "broadleaf_deciduous_boreal_shrub",
             "c3_arctic_grass",
             "c3_non-arctic_grass",
             "c4_grass",
             "unmanaged_c3_crop",
             "unmanaged_c3_irrigated",
             "temperate_corn",
             "irrigated_temperate_corn",
             "spring_wheat",
             "irrigated_spring_wheat",
             "winter_wheat",
             "irrigated_winter_wheat",
             "soybean",
             "irrigated_soybean",
             "barley",
             "irrigated_barley",
             "winter_barley",
             "irrigated_winter_barley",
             "rye",
             "irrigated_rye",
             "winter_rye",
             "irrigated_winter_rye",
             "cassava",
             "irrigated_cassava",
             "citrus",
             "irrigated_citrus",
             "cocoa",
             "irrigated_cocoa",
             "coffee",
             "irrigated_coffee",
             "cotton",
             "irrigated_cotton",
             "datepalm",
             "irrigated_datepalm",
             "foddergrass",
             "irrigated_foddergrass",
             "grapes",
             "irrigated_grapes",
             "groundnuts",
             "irrigated_groundnuts",
             "millet",
             "irrigated_millet",
             "oilpalm",
             "irrigated_oilpalm",
             "potatoes",
             "irrigated_potatoes",
             "pulses",
             "irrigated_pulses",
             "rapeseed",
             "irrigated_rapeseed",
             "rice",
             "irrigated_rice",
             "sorghum",
             "irrigated_sorghum",
             "sugarbeet",
             "irrigated_sugarbeet",
             "sugarcane",
             "irrigated_sugarcane",
             "sunflower",
             "irrigated_sunflower",
             "miscanthus",
             "irrigated_miscanthus",
             "switchgrass",
             "irrigated_switchgrass",
             "tropical_corn",
             "irrigated_tropical_corn",
             "tropical_soybean",
             "irrigated_tropical_soybean"]


# %% Import dataset

# Get list of all files in $indir matching $pattern
indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/"
pattern = "*h1.*-01-01-00000.nc"
filelist = glob.glob(indir + pattern)

# Set up function to drop unwanted vars in preprocessing of open_mfdataset()
def mfdataset_preproc(ds):
    vars_to_import = list(ds.dims) + \
        ["CPHASE", 
        "GDDHARV", 
        "GDDPLANT", 
        "GPP", 
        "GRAINC_TO_FOOD", 
        "NPP", 
        "TLAI", 
        "TOTVEGC", 
        "pfts1d_itype_veg",
        "pfts1d_ixy",
        "pfts1d_jxy",
        "pfts1d_lon",
        "pfts1d_lat"]
    varlist = list(ds.variables)
    vars_to_drop = list(np.setdiff1d(varlist, vars_to_import))
    ds = ds.drop_vars(vars_to_drop)
    ds = xr.decode_cf(ds, decode_times = True)
    return ds

# Import
this_ds = xr.open_mfdataset(filelist, \
    concat_dim="time", 
    preprocess=mfdataset_preproc)
# this_ds = utils.time_set_mid(this_ds, 'time')

# Get dates in a format that matplotlib can use
with warnings.catch_warnings():
    # Ignore this warning in this with-block
    warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar, 'noleap', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.")
    datetime_vals = this_ds.indexes["time"].to_datetimeindex()

# Get PFT list, integers (use only first timestep)
vegtype_int = this_ds.pfts1d_itype_veg
vegtype_int.values = vegtype_int.values.astype(int)
if not all((vegtype_int.values == vegtype_int.values[0,:]).all(axis=1)):
    raise ValueError("Some veg type changes over time")
vegtype_int = vegtype_int[0,:]

# Get PFT list, strings
vegtype_str = list(np.array(pftname)[vegtype_int.values])

# %% Read variable

# Which variable?
thisVar = "CPHASE"

def is_this_mgd_crop(x):
    notcrop_list = ["tree", "grass", "shrub", "unmanaged", "not_vegetated"]
    return not any(n in x for n in notcrop_list)
def get_thisVar_da(thisVar, this_ds, vegtype_str):
    # Make DataArray for this variable
    thisvar_da = np.array(this_ds.variables[thisVar])
    theseDims = this_ds.variables[thisVar].dims
    thisvar_da = xr.DataArray(thisvar_da, 
        dims = theseDims)

    # Define coordinates of this variable's DataArray
    dimsDict = dict()
    for thisDim in theseDims:
        if thisDim == "pft":
            dimsDict[thisDim] = vegtype_str
        elif any(np.array(list(this_ds.dims.keys())) == thisDim):
                dimsDict[thisDim] = this_ds[thisDim]
        else:
            raise ValueError("Unknown dimension for coordinate assignment: " + thisDim)
    thisvar_da = thisvar_da.assign_coords(dimsDict)

    # If it has PFT dimension, trim to managed crops
    if any(np.array(list(thisvar_da.dims)) == "pft"):
        is_crop = [ is_this_mgd_crop(x) for x in thisvar_da.pft.values ]
        thisvar_da = thisvar_da[:, is_crop]

    return thisvar_da

thisvar_da = get_thisVar_da(thisVar, this_ds, vegtype_str)


# %% Grid variable (takes a while) and make map

# ixy = get_thisVar_da("pfts1d_ixy", this_ds, vegtype_str)
# jxy = get_thisVar_da("pfts1d_jxy", this_ds, vegtype_str)
# lon = get_thisVar_da("lon", this_ds, vegtype_str)
# lat = get_thisVar_da("lat", this_ds, vegtype_str)
# ttime = get_thisVar_da("time", this_ds, vegtype_str)
ixy = this_ds.pfts1d_ixy
jxy = this_ds.pfts1d_jxy
lon = this_ds.lon
lat = this_ds.lat
ttime = this_ds.time

nlat = len(lat.values)
nlon = len(lon.values)
npft = np.max(vegtype_int.values) + 1
ntim = len(ttime.values)

tmp_tpyx = np.empty([ntim, npft, nlat, nlon])
tmp_tpyx[:, \
    vegtype_int.values, 
    jxy.values.astype(int) - 1, 
    ixy.values.astype(int) - 1] = this_ds.variables[thisVar].values

tmp2_tpyx = xr.DataArray(tmp_tpyx, dims=("time","pft","lat","lon"))
tmp2_tpyx = tmp2_tpyx.assign_coords( \
    time=ttime,
    pft=pftname,
    lat=lat.values,
    lon=lon.values)
tmp2_tpyx.name = thisVar
is_crop = [ is_this_mgd_crop(x) for x in tmp2_tpyx.pft.values ]
tmp2_tpyx = tmp2_tpyx[:, is_crop]
print(tmp2_tpyx)

# Make map
tmp3 = tmp2_tpyx.isel(time=0, pft=0)
tmp4 = cyclic_dataarray(tmp3)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolor(tmp4.lon.values, tmp4.lat.values, tmp4, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()


# %% Plot and make map, more efficiently

tmp = thisvar_da[dict(time=0)]

# ixy = this_ds.pfts1d_ixy[dict(time=0)]
# jxy = this_ds.pfts1d_jxy[dict(time=0)]
ixy_da = get_thisVar_da("pfts1d_ixy", this_ds, vegtype_str)
jxy_da = get_thisVar_da("pfts1d_jxy", this_ds, vegtype_str)
ixy = ixy_da[dict(time=0)]
jxy = jxy_da[dict(time=0)]
lon = this_ds.lon
lat = this_ds.lat

vt_da = get_thisVar_da("pfts1d_itype_veg", this_ds, vegtype_str)

vt = vt_da[dict(time=0)].values

nlat = len(lat.values)
nlon = len(lon.values)
npft = np.max(vegtype_int.values) + 1

tmp_pyx = np.empty([npft, nlat, nlon])
tmp_pyx[vt, 
    jxy.values.astype(int) - 1, 
    ixy.values.astype(int) - 1] = tmp.values

tmp2_pyx = xr.DataArray(tmp_pyx, dims=("pft","lat","lon"))
tmp2_pyx = tmp2_pyx.assign_coords( \
    pft=pftname,
    lat=lat.values,
    lon=lon.values)
tmp2_pyx.name = thisVar
is_crop = [ is_this_mgd_crop(x) for x in tmp2_pyx.pft.values ]
tmp2_pyx = tmp2_pyx[is_crop]

# Make map
tmp3 = tmp2_pyx.isel(pft=0)
tmp4 = cyclic_dataarray(tmp3)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolor(tmp4.lon.values, tmp4.lat.values, tmp4, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()


# %% Plot and make map, more efficiently, as function

def grid_one_timestep(thisvar_da, time_index):

    # Get this variable's values for this time step
    thisvar_da_1time = thisvar_da[dict(time=time_index)]

    # Get gridcell indices for this time step
    ixy_da = get_thisVar_da("pfts1d_ixy", this_ds, vegtype_str)
    jxy_da = get_thisVar_da("pfts1d_jxy", this_ds, vegtype_str)
    ixy = ixy_da[dict(time=time_index)]
    jxy = jxy_da[dict(time=time_index)]

    # Get PFT indices for this time step
    vt_da = get_thisVar_da("pfts1d_itype_veg", this_ds, vegtype_str)
    vt = vt_da[dict(time=time_index)].values
    
    # Get dataset lon/lat grid
    lon = this_ds.lon
    lat = this_ds.lat

    # Set up empty array: PFT * lat * lon
    npft = np.max(vegtype_int.values) + 1
    nlat = len(lat.values)
    nlon = len(lon.values)
    thisvar_pyx = np.empty([npft, nlat, nlon])

    # Fill with this variable
    thisvar_pyx[vt, 
        jxy.values.astype(int) - 1, 
        ixy.values.astype(int) - 1] = thisvar_da_1time.values

    # Assign coordinates and name
    thisvar_pyx = xr.DataArray(thisvar_pyx, dims=("pft","lat","lon"))
    thisvar_pyx = thisvar_pyx.assign_coords( \
        pft=pftname,
        lat=lat.values,
        lon=lon.values)
    thisvar_pyx.name = thisVar

    # Restrict to managed crops
    is_crop = [ is_this_mgd_crop(x) for x in thisvar_pyx.pft.values ]
    thisvar_pyx = thisvar_pyx[is_crop]

    return thisvar_pyx

def grid_timeslice(thisvar_da, time_str_0: str, time_str_1: str = ""):

    one_timestep = time_str_1 == ""
    if (one_timestep):
        time_slice = slice(time_str_0)
    else:
        time_slice = slice(time_str_0, time_str_1)

    # Get this variable's values for this time slice
    thisvar_da_1time = thisvar_da[dict(time=time_slice)]

    # Get gridcell indices for this time slice
    ixy_da = get_thisVar_da("pfts1d_ixy", this_ds, vegtype_str)
    jxy_da = get_thisVar_da("pfts1d_jxy", this_ds, vegtype_str)
    ixy = ixy_da[dict(time=time_slice)]
    jxy = jxy_da[dict(time=time_slice)]

    # Get PFT indices for this time slice
    vt_da = get_thisVar_da("pfts1d_itype_veg", this_ds, vegtype_str)
    vt = vt_da[dict(time=time_slice)].values
    
    # Get dataset lon/lat grid
    lon = this_ds.lon
    lat = this_ds.lat

    # Set up empty array: PFT * lat * lon
    npft = np.max(vegtype_int.values) + 1
    nlat = len(lat.values)
    nlon = len(lon.values)
    if (one_timestep):
        raise ValueError("Finish coding this")
        ntim = len(ttime.values)
        tmp_tpyx = np.empty([ntim, npft, nlat, nlon])
    else:
        thisvar_out = np.empty([npft, nlat, nlon])

    # Fill with this variable
    if (one_timestep):
        raise ValueError("Finish coding this")
    else:
        thisvar_out[vt, 
            jxy.values.astype(int) - 1, 
            ixy.values.astype(int) - 1] = thisvar_da_1time.values

    # Assign coordinates and name
    if (one_timestep):
        raise ValueError("Finish coding this")
    else:
        thisvar_out = xr.DataArray(thisvar_out, dims=("pft","lat","lon"))
        thisvar_out = thisvar_out.assign_coords( \
            pft=pftname,
            lat=lat.values,
            lon=lon.values)
    thisvar_out.name = thisVar

    # Restrict to managed crops
    is_crop = [ is_this_mgd_crop(x) for x in thisvar_out.pft.values ]
    thisvar_out = thisvar_out[is_crop]

    return thisvar_out

# Grid this timestep
tmp_pyx = grid_one_timestep(thisvar_da, 0)

# Make map
tmp_yx = tmp_pyx.isel(pft=0)
tmp_yx = cyclic_dataarray(tmp_yx)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolor(tmp_yx.lon.values, tmp_yx.lat.values, tmp_yx, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()
