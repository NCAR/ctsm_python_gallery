"""utility functions"""
"""copied from klindsay, https://github.com/klindsay28/CESM2_coup_carb_cycle_JAMES/blob/master/utils.py"""

import re
import cf_units as cf
import cftime
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point

#from xr_ds_ex import xr_ds_ex

# generate annual means, weighted by days / month
def weighted_annual_mean(array):
    mon_day  = xr.DataArray(np.array([31,28,31,30,31,30,31,31,30,31,30,31]), dims=['month'])
    mon_wgt  = mon_day/mon_day.sum()
    return (array.rolling(time=12, center=False) # rolling
            .construct("month") # construct the array
            .isel(time=slice(11, None, 12)) # slice so that the first element is [1..12], second is [13..24]
            .dot(mon_wgt, dims=["month"]))

def change_units(ds, variable_str, variable_bounds_str, target_unit_str):
    """ Applies unit conversion on an xarray DataArray """
    orig_units = cf.Unit(ds[variable_str].attrs["units"])
    target_units = cf.Unit(target_unit_str)
    variable_in_new_units = xr.apply_ufunc(
        orig_units.convert,
        ds[variable_bounds_str],
        target_units,
        output_dtypes=[ds[variable_bounds_str].dtype],
    )
    return variable_in_new_units

def clean_units(units):
    """replace some troublesome unit terms with acceptable replacements"""
    replacements = {'kgC':'kg', 'gC':'g', 'gC13':'g', 'gC14':'g', 'gN':'g',
                    'unitless':'1',
                    'years':'common_years', 'yr':'common_year',
                    'meq':'mmol', 'neq':'nmol'}
    units_split = re.split('( |\(|\)|\^|\*|/|-[0-9]+|[0-9]+)', units)
    units_split_repl = \
        [replacements[token] if token in replacements else token for token in units_split]
    return ''.join(units_split_repl)

def copy_fill_settings(da_in, da_out):
    """
    propagate _FillValue and missing_value settings from da_in to da_out
    return da_out
    """
    if '_FillValue' in da_in.encoding:
        da_out.encoding['_FillValue'] = da_in.encoding['_FillValue']
    else:
        da_out.encoding['_FillValue'] = None
    if 'missing_value' in da_in.encoding:
        da_out.attrs['missing_value'] = da_in.encoding['missing_value']
    return da_out

def dim_cnt_check(ds, varname, dim_cnt):
    """confirm that varname in ds has dim_cnt dimensions"""
    if len(ds[varname].dims) != dim_cnt:
        msg_full = 'unexpected dim_cnt=%d, varname=%s' % (len(ds[varname].dims), varname)
        raise ValueError(msg_full)

def time_set_mid(ds, time_name):
    """
    set ds[time_name] to midpoint of ds[time_name].attrs['bounds'], if bounds attribute exists
    type of ds[time_name] is not changed
    ds is returned
    """

    if 'bounds' not in ds[time_name].attrs:
        return ds

    # determine units and calendar of unencoded time values
    if ds[time_name].dtype == np.dtype('O'):
        units = 'days since 0000-01-01'
        calendar = 'noleap'
    else:
        units = ds[time_name].attrs['units']
        calendar = ds[time_name].attrs['calendar']

    # construct unencoded midpoint values, assumes bounds dim is 2nd
    tb_name = ds[time_name].attrs['bounds']
    if ds[tb_name].dtype == np.dtype('O'):
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
    else:
        tb_vals = ds[tb_name].values
    tb_mid = tb_vals.mean(axis=1)

    # set ds[time_name] to tb_mid
    if ds[time_name].dtype == np.dtype('O'):
        ds[time_name] = cftime.num2date(tb_mid, units=units, calendar=calendar)
    else:
        ds[time_name] = tb_mid

    return ds

def time_year_plus_frac(ds, time_name):
    """return time variable, as year plus fraction of year"""

    # this is straightforward if time has units='days since 0000-01-01' and calendar='noleap'
    # so convert specification of time to that representation

    # get time values as an np.ndarray of cftime objects
    if np.dtype(ds[time_name]) == np.dtype('O'):
        tvals_cftime = ds[time_name].values
    else:
        tvals_cftime = cftime.num2date(
            ds[time_name].values, ds[time_name].attrs['units'], ds[time_name].attrs['calendar'])

    # convert cftime objects to representation mentioned above
    tvals_days = cftime.date2num(tvals_cftime, 'days since 0000-01-01', calendar='noleap')

    return tvals_days / 365.0


# add cyclic point
def cyclic_dataarray(da, coord='lon'):
    """ Add a cyclic coordinate point to a DataArray along a specified
    named coordinate dimension.
    >>> from xray import DataArray
    >>> data = DataArray([[1, 2, 3], [4, 5, 6]],
    ...                      coords={'x': [1, 2], 'y': range(3)},
    ...                      dims=['x', 'y'])
    >>> cd = cyclic_dataarray(data, 'y')
    >>> print cd.data
    array([[1, 2, 3, 1],
           [4, 5, 6, 4]])
    """
    assert isinstance(da, xr.DataArray)

    lon_idx = da.dims.index(coord)
    cyclic_data, cyclic_coord = add_cyclic_point(da.values,
                                                 coord=da.coords[coord],
                                                 axis=lon_idx)

    # Copy and add the cyclic coordinate and data
    new_coords = dict(da.coords)
    new_coords[coord] = cyclic_coord
    new_values = cyclic_data

    new_da = xr.DataArray(new_values, dims=da.dims, coords=new_coords)

    # Copy the attributes for the re-constructed data and coords
    for att, val in da.attrs.items():
        new_da.attrs[att] = val
    for c in da.coords:
        for att in da.coords[c].attrs:
            new_da.coords[c].attrs[att] = da.coords[c].attrs[att]

    return new_da

# as above, but for a dataset
# doesn't work because dims are locked in a dataset
'''
def cyclic_dataset(ds, coord='lon'):
    assert isinstance(ds, xr.Dataset)

    lon_idx = ds.dims.index(coord)
    cyclic_data, cyclic_coord = add_cyclic_point(ds.values,
                                                 coord=ds.coords[coord],
                                                 axis=lon_idx)

    # Copy and add the cyclic coordinate and data
    new_coords = dict(ds.coords)
    new_coords[coord] = cyclic_coord
    new_values = cyclic_data

    new_ds = xr.DataSet(new_values, dims=ds.dims, coords=new_coords)

    # Copy the attributes for the re-constructed data and coords
    for att, val in ds.attrs.items():
        new_ds.attrs[att] = val
    for c in ds.coords:
        for att in ds.coords[c].attrs:
            new_ds.coords[c].attrs[att] = ds.coords[c].attrs[att]

    return new_ds
'''


# List of PFTs used in CLM
def define_pftlist():
    pftlist =  ["not_vegetated",
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
    return pftlist


# Does this vegetation type's name match (for a given comparison method) any member of a filtering list?
'''
Methods:
    ok_contains:    True if any member of this_filter is found in this_vegtype.
    notok_contains: True of no member of this_filter is found in this_vegtype.
    ok_exact:       True if this_vegtype matches any member of this_filter 
                    exactly.
    notok_exact:    True if this_vegtype does not match any member of 
                    this_filter exactly.
'''
def is_this_vegtype(this_vegtype, this_filter, this_method):

    # Make sure data type of this_vegtype is acceptable
    data_type_ok = lambda x: isinstance(x, str) or isinstance(x, int) or isinstance(x, np.int64)
    ok_input = True
    if not data_type_ok(this_vegtype):
        if isinstance(this_vegtype, xr.core.dataarray.DataArray):
            this_vegtype = this_vegtype.values
        if isinstance(this_vegtype, (list, np.ndarray)):
            if len(this_vegtype) == 1 and data_type_ok(this_vegtype[0]):
                this_vegtype = this_vegtype[0]
            elif data_type_ok(this_vegtype[0]):
                raise TypeError("is_this_vegtype(): this_vegtype must be a single string or integer, not a list of them. Did you mean to call is_each_vegtype() instead?")
            else:
                ok_input = False
        else:
            ok_input = False
    if not ok_input:
        raise TypeError(f"is_this_vegtype(): First argument (this_vegtype) must be a string or integer, not {type(this_vegtype)}")
    
    # Make sure data type of this_filter is acceptable
    if not np.iterable(this_filter):
        raise TypeError(f"is_this_vegtype(): Second argument (this_filter) must be iterable (e.g., a list), not {type(this_filter)}")
    
    # Perform the comparison
    if this_method == "ok_contains":
        return any(n in this_vegtype for n in this_filter)
    elif this_method == "notok_contains":
        return not any(n in this_vegtype for n in this_filter)
    elif this_method == "ok_exact":
        return any(n == this_vegtype for n in this_filter)
    elif this_method == "notok_exact":
        return not any(n == this_vegtype for n in this_filter)
    else:
        raise ValueError(f"Unknown comparison method: '{this_method}'")


# Get boolean list of whether each vegetation type in list is a managed crop
'''
    this_vegtypelist: The list of vegetation types whose members you want to 
                      test.
    this_filter:      The list of strings against which you want to compare 
                      each member of this_vegtypelist.
    this_method:      How you want to do the comparison. See is_this_vegtype().
'''
def is_each_vegtype(this_vegtypelist, this_filter, this_method):
    return [is_this_vegtype(x, this_filter, this_method) for x in this_vegtypelist]


# Convert a latitude axis that's -180 to 180 around the international date line to one that's 0 to 360 around the prime meridian. If you pass in a Dataset or DataArray, the "lon" coordinates will be changed. Otherwise, it assumes you're passing in numeric data.
def lon_idl2pm(lons_in):
    def do_it(tmp):
        if np.any(tmp > 180):
            raise ValueError(f"Maximum longitude is already > 180 ({np.max(tmp)})")
        elif np.any(tmp < -180):
            raise ValueError(f"Minimum longitude is < -180 ({np.min(tmp)})")
        tmp = tmp + 360
        tmp = np.mod(tmp, 360)
        return tmp
    if isinstance(lons_in, (xr.DataArray, xr.Dataset)):
        lons_out = lons_in
        lons_out['lon'] = do_it(lons_in.lon.values)
    else:
        lons_out = do_it(lons_in)
        
    return lons_out


# Convert a latitude axis that's 0 to 360 around the prime meridian to one that's -180 to 180 around the international date line. If you pass in a Dataset or DataArray, the "lon" coordinates will be changed. Otherwise, this assumes you're passing in numeric data.
def lon_pm2idl(lons_in):
    def do_it(tmp):
        if np.any(tmp < 0):
            raise ValueError(f"Minimum longitude is already < 0 ({np.min(tmp)})")
        elif np.any(tmp > 360):
            raise ValueError(f"Maximum longitude is > 360 ({np.max(tmp)})")
        tmp = np.mod((tmp + 180),360)-180
        return tmp
    if isinstance(lons_in, (xr.DataArray, xr.Dataset)):
        lons_out = lons_in
        lons_out['lon'] = do_it(lons_in.lon.values)
    else:
        lons_out = do_it(lons_in)
        
    return lons_out


# List (strings) of managed crops in CLM.
def define_mgdcrop_list():
    notcrop_list = ["tree", "grass", "shrub", "unmanaged", "not_vegetated"]
    defined_pftlist = define_pftlist()
    is_crop = is_each_vegtype(defined_pftlist, notcrop_list, "notok_contains")
    return [defined_pftlist[i] for i, x in enumerate(is_crop) if x]


# Convert list of vegtype strings to integer index equivalents.
def vegtype_str2int(vegtype_str, vegtype_mainlist=None):
    if isinstance(vegtype_mainlist, xr.Dataset):
        vegtype_mainlist = vegtype_mainlist.vegtype_str.values
    elif isinstance(vegtype_mainlist, xr.DataArray):
        vegtype_mainlist = vegtype_mainlist.values
    elif vegtype_mainlist == None:
        vegtype_mainlist = define_pftlist()
    if not isinstance(vegtype_mainlist, list) and isinstance(vegtype_mainlist[0], str):
        if isinstance(vegtype_mainlist, list):
            raise TypeError(f"Not sure how to handle vegtype_mainlist as list of {type(vegtype_mainlist[0])}")
        else:
            raise TypeError(f"Not sure how to handle vegtype_mainlist as type {type(vegtype_mainlist[0])}")
    ind_dict = dict((k,i) for i,k in enumerate(vegtype_mainlist))
    inter = set(ind_dict).intersection(vegtype_str)
    indices = [ ind_dict[x] for x in inter ]
    return indices

# Check the type of a selection. Used in xr_flexsel(). This function ended up only being used once there, but keep it separate anyway to avoid having to re-do it in the future.
def check_sel_type(this_sel):
    if isinstance(this_sel, slice):
        if this_sel == slice(0):
            raise ValueError("slice(0) will be empty")
        elif this_sel.start != None:
            return type(this_sel.start)
        elif this_sel.stop != None:
            return type(this_sel.stop)
        elif this_sel.step != None:
            return type(this_sel.step)
        else:
            raise TypeError("slice is all None?")
    else:
        return type(this_sel)


# Flexibly subset time(s) and/or vegetation type(s) from an xarray Dataset or DataArray. Keyword arguments like dimension=selection. Selections can be individual values or slice()s. Optimize memory usage by beginning keyword argument list with the selections that will result in the largest reduction of object size.
def xr_flexsel(xr_object, patches1d_itype_veg=None, unsupported=False, **kwargs):
    
    # For now, only time and vegtype selections are supported
    if not unsupported:
        for key in kwargs.keys():
            if key not in ["time", "vegtype"]:
                raise ValueError(f"xr_flexsel() only tested with time and vegtype. To run with unsupported dimensions like {key}, specify unsupported=True.")
    
    for key, value in kwargs.items():

        if key == "vegtype":

            # Convert to list, if needed
            if not isinstance(value, list):
                value = [value]
            
            # Convert to indices, if needed
            if isinstance(value[0], str):
                value = vegtype_str2int(value)
            
            # Get list of boolean(s)
            if isinstance(value[0], int):
                if isinstance(patches1d_itype_veg, type(None)):
                    patches1d_itype_veg = xr_object.patches1d_itype_veg.values
                elif isinstance(patches1d_itype_veg, xr.core.dataarray.DataArray):
                    patches1d_itype_veg = patches1d_itype_veg.values
                is_vegtype = is_each_vegtype(patches1d_itype_veg, value, "ok_exact")
            elif isinstance(value[0], bool):
                if len(value) != len(xr_object.patch):
                    raise ValueError(f"If providing boolean 'vegtype' argument to xr_flexsel(), it must be the same length as xr_object.patch ({len(value)} vs. {len(xr_object.patch)})")
                is_vegtype = value
            else:
                raise TypeError(f"Not sure how to handle 'vegtype' of type {type(value)}")
            xr_object = xr_object.isel(patch=[i for i, x in enumerate(is_vegtype) if x])
            if "ivt" in xr_object:
                xr_object = xr_object.isel(ivt=is_each_vegtype(xr_object.ivt.values, value, "ok_exact"))
        
        else:
            this_type = check_sel_type(value)
            if this_type == int:
                # Have to select like this instead of with index directly because otherwise assign_coords() will throw an error. Not sure why.
                if isinstance(value, int):
                    value = slice(value,value+1)
                xr_object = xr_object.isel({key: value})
            elif this_type == str:
                xr_object = xr_object.sel({key: value})
            else:
                raise TypeError(f"Selection must be type int, str, or slice of those (not {type(value)})")
    
    return xr_object


# Get PFT of each patch, in both integer and string forms.
def ivt_int_str(this_ds, this_pftlist):
    # First, get all the integer values; should be time*pft or pft*time. We will eventually just take the first timestep.
    vegtype_int = this_ds.patches1d_itype_veg
    vegtype_int.values = vegtype_int.values.astype(int)

    # Convert to strings.
    vegtype_str = list(np.array(this_pftlist)[vegtype_int.values])

    # Return a dictionary with both results
    return {"int": vegtype_int, "str": vegtype_str, "all_str": this_pftlist}


# Convert a list of strings with vegetation type names into a DataArray. Used to add vegetation type info in import_ds().
def get_vegtype_str_da(vegtype_str):
    nvt = len(vegtype_str)
    thisName = "vegtype_str"
    vegtype_str_da = xr.DataArray(\
        vegtype_str, 
        coords={"ivt": np.arange(0,nvt)}, 
        dims=["ivt"],
        name = thisName)
    return vegtype_str_da


# Function to drop unwanted variables in preprocessing of open_mfdataset(), making sure to NOT drop any unspecified variables that will be useful in gridding. Also adds vegetation type info in the form of a DataArray of strings.
# Also renames "pft" dimension (and all like-named variables, e.g., pft1d_itype_veg_str) to be named like "patch". This can later be reversed, for compatibility with other code, using patch2pft().
def mfdataset_preproc(ds, vars_to_import, vegtypes_to_import):

    if vars_to_import != None:
        # Get list of dimensions present in variables in vars_to_import.
        dimList = []
        for thisVar in vars_to_import:
            # list(set(x)) returns a list of the unique items in x
            dimList = list(set(dimList + list(ds.variables[thisVar].dims)))
        
        # Get any _1d variables that are associated with those dimensions. These will be useful in gridding. Also, if any dimension is "pft", set up to rename it and all like-named variables to "patch"
        onedVars = []
        pft2patch_dict = {}
        for thisDim in dimList:
            pattern = re.compile(f"{thisDim}.*1d")
            matches = [x for x in list(ds.keys()) if pattern.search(x) != None]
            onedVars = list(set(onedVars + matches))
            if thisDim == "pft":
                pft2patch_dict["pft"] = "patch"
                for m in matches:
                    pft2patch_dict[m] = m.replace("pft","patch").replace("patchs","patches")
        
        # Add dimensions and _1d variables to vars_to_import
        vars_to_import = list(set(vars_to_import \
            + list(ds.dims) + onedVars))

        # Get list of variables to drop
        varlist = list(ds.variables)
        vars_to_drop = list(np.setdiff1d(varlist, vars_to_import))

        # Drop them
        ds = ds.drop_vars(vars_to_drop)

    # Rename "pft" dimension and variables to "patch", if needed
    if len(pft2patch_dict) > 0:
        ds = ds.rename(pft2patch_dict)
    
    # Add vegetation type info
    if "patches1d_itype_veg" in list(ds):
        this_pftlist = define_pftlist()
        ivt_int_str(ds, this_pftlist) # Includes check of whether vegtype changes over time anywhere
        vegtype_da = get_vegtype_str_da(this_pftlist)
        patches1d_itype_veg_str = vegtype_da.values[ds.isel(time=0).patches1d_itype_veg.values.astype(int)]
        npatch = len(patches1d_itype_veg_str)
        patches1d_itype_veg_str = xr.DataArray( \
            patches1d_itype_veg_str,
            coords={"patch": np.arange(0,npatch)}, 
            dims=["patch"],
            name = "patches1d_itype_veg_str")
        ds = xr.merge([ds, vegtype_da, patches1d_itype_veg_str])

    # Restrict to veg. types of interest, if any
    if vegtypes_to_import != None:
        ds = xr_flexsel(ds, vegtype=vegtypes_to_import)

    # Finish import
    ds = xr.decode_cf(ds, decode_times = True)
    return ds


# Rename "patch" dimension and any associated variables back to "pft". Uses a dictionary with the names of the dimensions and variables we want to rename. This allows us to do it all at once, which may be more efficient than one-by-one.
def patch2pft(xr_object):

    # Rename "patch" dimension
    patch2pft_dict = {}
    for thisDim in xr_object.dims:
        if thisDim == "patch":
            patch2pft_dict["patch"] = "pft"
            break
    
    # Rename variables containing "patch"
    if isinstance(xr_object, xr.Dataset):
        pattern = re.compile("patch.*1d")
        matches = [x for x in list(xr_object.keys()) if pattern.search(x) != None]
        if len(matches) > 0:
            for m in matches:
                patch2pft_dict[m] = m.replace("patches","patchs").replace("patch","pft")
    
    # Do the rename
    if len(patch2pft_dict) > 0:
        xr_object = xr_object.rename(patch2pft_dict)
    
    return xr_object


# Import a dataset that can be spread over multiple files, only including specified variables and/or vegetation types, concatenating by time. DOES actually read the dataset into memory, but only AFTER dropping unwanted variables and/or vegetation types.
def import_ds(filelist, myVars=None, myVegtypes=None):

    # Convert myVegtypes here, if needed, to avoid repeating the process each time you read a file in xr.open_mfdataset().
    if myVegtypes != None:
        if not isinstance(myVegtypes, list):
            myVegtypes = [myVegtypes]
        if isinstance(myVegtypes[0], str):
            myVegtypes = vegtype_str2int(myVegtypes)

    # The xarray open_mfdataset() "preprocess" argument requires a function that takes exactly one variable (an xarray.Dataset object). Wrapping mfdataset_preproc() in this lambda function allows this. Could also just allow mfdataset_preproc() to access myVars and myVegtypes directly, but that's bad practice as it could lead to scoping issues.
    mfdataset_preproc_closure = \
        lambda ds: mfdataset_preproc(ds, myVars, myVegtypes)

    # Import
    if isinstance(filelist, list):
        this_ds = xr.open_mfdataset(sorted(filelist), \
            data_vars="minimal", 
            preprocess=mfdataset_preproc_closure)
    elif isinstance(filelist, str):
        this_ds = xr.open_dataset(filelist)
        this_ds = mfdataset_preproc(this_ds, myVars, myVegtypes)
        this_ds = this_ds.compute()
    
    return this_ds


# Return a DataArray, with defined coordinates, for a given variable in a dataset.
def get_thisVar_da(thisVar, this_ds):

    # Make DataArray for this variable
    thisvar_da = np.array(this_ds.variables[thisVar])
    theseDims = this_ds.variables[thisVar].dims
    thisvar_da = xr.DataArray(thisvar_da, 
        dims = theseDims)

    # Define coordinates of this variable's DataArray
    dimsDict = dict()
    for thisDim in theseDims:
        dimsDict[thisDim] = this_ds[thisDim]
    thisvar_da = thisvar_da.assign_coords(dimsDict)

    return thisvar_da


# Given a DataArray, remove all patches except those planted with managed crops.
def trim_da_to_mgd_crop(thisvar_da, patches1d_itype_veg_str):

    # Handle input DataArray without patch dimension
    if not any(np.array(list(thisvar_da.dims)) == "patch"):
        print("Input DataArray has no patch dimension and therefore trim_to_mgd_crop() has no effect.")
        return thisvar_da
    
    # Throw error if patches1d_itype_veg_str isn't strings
    if isinstance(patches1d_itype_veg_str, xr.DataArray):
        patches1d_itype_veg_str = patches1d_itype_veg_str.values
    if not isinstance(patches1d_itype_veg_str[0], str):
        raise TypeError("Input patches1d_itype_veg_str is not in string form, and therefore trim_to_mgd_crop() cannot work.")
    
    # Get boolean list of whether each patch is planted with a managed crop
    notcrop_list = ["tree", "grass", "shrub", "unmanaged", "not_vegetated"]
    is_crop = is_each_vegtype(patches1d_itype_veg_str, notcrop_list, "notok_contains")

    # Warn if no managed crops were found, but still return the empty result
    if np.all(np.bitwise_not(is_crop)):
        print("No managed crops found! Returning empty DataArray.")
    return thisvar_da.isel(patch = [i for i, x in enumerate(is_crop) if x])


# Make a geographically gridded DataArray (with dimensions time, vegetation type [as string], lat, lon) of one variable within a Dataset. Optional keyword arguments will be passed to xr_flexsel() to select single steps or slices along the specified ax(ie)s.
# SSR TODO: IN PROGRESS: Allow for flexible input and output dimensions.
def grid_one_variable(this_ds, thisVar, unsupported=False, **kwargs):
    
    # Get this Dataset's values for selection(s), if provided
    this_ds = xr_flexsel(this_ds, \
        unsupported=unsupported,
        **kwargs)
    
    # Get DataArrays needed for gridding
    thisvar_da = get_thisVar_da(thisVar, this_ds)
    ixy_da = get_thisVar_da("patches1d_ixy", this_ds)
    jxy_da = get_thisVar_da("patches1d_jxy", this_ds)
    vt_da = get_thisVar_da("patches1d_itype_veg", this_ds)
    
    # Renumber vt_da to work as indices on new ivt dimension, if needed.
    ### Ensures that the unique set of vt_da values begins with 1 and
    ### contains no missing steps.
    if "vegtype" in kwargs.keys():
        vt_da.values = np.array([np.where(this_ds.ivt.values == x)[0][0] for x in vt_da.values])
    
    # Get new dimension list
    new_dims = list(thisvar_da.dims)
    ### Replace "patch" with "ivt_str" (vegetation type, as string)
    if "patch" in new_dims:
        new_dims = ["ivt_str" if x == "patch" else x for x in new_dims]
    ### Add lat and lon to end of list
    new_dims = new_dims + ["lat", "lon"]

    # Set up empty array
    n_list = []
    for dim in new_dims:
        if dim == "ivt_str":
            n = this_ds.sizes["ivt"]
        elif dim in thisvar_da.coords:
            n = thisvar_da.sizes[dim]
        else:
            n = this_ds.sizes[dim]
        n_list = n_list + [n]
    thisvar_gridded = np.empty(n_list)

    # Fill with this variable
    if new_dims != ['time', 'ivt_str', 'lat', 'lon']:
        raise ValueError("For now, grid_one_variable() only works with output dimensions ['time', 'ivt_str', 'lat', 'lon']")
    thisvar_gridded[:,
        vt_da, 
        jxy_da.values.astype(int) - 1, 
        ixy_da.values.astype(int) - 1] = thisvar_da.values

    # Assign coordinates and name
    thisvar_gridded = xr.DataArray(thisvar_gridded, dims=tuple(new_dims))
    for dim in new_dims:
        if dim == "ivt_str":
            values = this_ds.vegtype_str.values
        elif dim in thisvar_da.coords:
            values = thisvar_da[dim]
        else:
            values = this_ds[dim].values
        thisvar_gridded = thisvar_gridded.assign_coords({dim: values})
    thisvar_gridded.name = thisVar

    return thisvar_gridded

