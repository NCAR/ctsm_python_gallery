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


# Import a dataset that's spread over multiple files, only including specified variables. Concatenate by time.
def import_ds_from_filelist(filelist, myVars=None):

    # Set up function to drop unwanted vars in preprocessing of open_mfdataset(), making sure to include any unspecified variables that will be useful in gridding.
    def mfdataset_preproc(ds, vars_to_import):

        if vars_to_import != None:
            # Get list of dimensions present in variables in vars_to_import.
            dimList = []
            for thisVar in vars_to_import:
                # list(set(x)) returns a list of the unique items in x
                dimList = list(set(dimList + list(ds.variables[thisVar].dims)))
            
            # Get any _1d variables that are associated with those dimensions. These will be useful in gridding.
            onedVars = []
            for thisDim in dimList:
                pattern = re.compile(f"{thisDim}.*1d")
                matches = [x for x in list(ds.keys()) if pattern.search(x) != None]
                onedVars = list(set(onedVars + matches))
            
            # Add dimensions and _1d variables to vars_to_import
            vars_to_import = list(set(vars_to_import \
                + list(ds.dims) + onedVars))

            # Get list of variables to drop
            varlist = list(ds.variables)
            vars_to_drop = list(np.setdiff1d(varlist, vars_to_import))

            # Drop them
            ds = ds.drop_vars(vars_to_drop)

        # Finish import
        ds = xr.decode_cf(ds, decode_times = True)
        return ds

    # xr.open_mfdataset()'s "preprocess" argument requires a function that only takes one variable (an xarray.Dataset object). Wrapping mfdataset_preproc() in this lambda function allows this. Could also just allow mfdataset_preproc() to access the myVars directly, but that's bad practice as it could lead to scoping issues.
    mfdataset_preproc_closure = \
        lambda ds: mfdataset_preproc(ds, myVars)

    # Import
    this_ds = xr.open_mfdataset(filelist, \
        concat_dim="time", 
        preprocess=mfdataset_preproc_closure)
    
    return this_ds
