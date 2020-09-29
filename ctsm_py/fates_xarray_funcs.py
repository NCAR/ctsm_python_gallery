'''functions for using fates and xarray'''
import xarray as xr
import numpy as np

def scpf_to_scls_by_pft(scpf_var, dataset):
    """function to reshape a fates multiplexed size and pft-indexed variable to one indexed by size class and pft
    first argument should be an xarray DataArray that has the FATES SCPF dimension
    second argument should be an xarray Dataset that has the FATES SCLS dimension 
    (possibly the dataset encompassing the dataarray being transformed)
    returns an Xarray DataArray with the size and pft dimensions disentangled"""
    n_scls = len(dataset.fates_levscls)
    ds_out = (scpf_var.rolling(fates_levscpf=n_scls, center=False)
            .construct("fates_levscls")
            .isel(fates_levscpf=slice(n_scls-1, None, n_scls))
            .rename({'fates_levscpf':'fates_levpft'})
            .assign_coords({'fates_levscls':dataset.fates_levscls})
            .assign_coords({'fates_levpft':dataset.fates_levpft}))
    ds_out.attrs['long_name'] = scpf_var.attrs['long_name']
    ds_out.attrs['units'] = scpf_var.attrs['units']
    return(ds_out)


def scag_to_scls_by_age(scag_var, dataset):
    """function to reshape a fates multiplexed size and pft-indexed variable to one indexed by size class and pft                                                                                                      
    first argument should be an xarray DataArray that has the FATES SCAG dimension                                                                                                                                     
    second argument should be an xarray Dataset that has the FATES age dimension                                                                                                                                      
   (possibly the dataset encompassing the dataarray being transformed)                                                                                                                                                     returns an Xarray DataArray with the size and age dimensions disentangled"""
    n_scls = len(dataset.fates_levscls)
    ds_out = (scag_var.rolling(fates_levscag=n_scls, center=False)
            .construct("fates_levscls")
            .isel(fates_levscag=slice(n_scls-1, None, n_scls))
            .rename({'fates_levscag':'fates_levage'})
            .assign_coords({'fates_levscls':dataset.fates_levscls})
            .assign_coords({'fates_levage':dataset.fates_levage}))
    ds_out.attrs['long_name'] = scag_var.attrs['long_name']
    ds_out.attrs['units'] = scag_var.attrs['units']
    return(ds_out)



def monthly_to_annual(array):
    """ calculate annual mena from monthly data, using unequal month lengths fros noleap calendar.  
    originally written by Keith Lindsay."""
    mon_day  = xr.DataArray(np.array([31.,28.,31.,30.,31.,30.,31.,31.,30.,31.,30.,31.]), dims=['month'])
    mon_wgt  = mon_day/mon_day.sum()
    return (array.rolling(time=12, center=False) # rolling
            .construct("month") # construct the array
            .isel(time=slice(11, None, 12)) # slice so that the first element is [1..12], second is [13..24]
            .dot(mon_wgt, dims=["month"]))

def monthly_to_month_by_year(array):
    """ go from monthly data to month x year data (for calculating climatologies, etc"""
    return (array.rolling(time=12, center=False) # rolling
            .construct("month") # construct the array
            .isel(time=slice(11, None, 12)).rename({'time':'year'}))


