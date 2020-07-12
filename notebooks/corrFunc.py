import numpy as np
import xarray as xr
import bottleneck
from scipy import signal,stats

# also try weighting correlation based on monthly contribution to annual flux
# This is code that's likely better suited for a 3rd partly library like esmlab
# TODO, file issue with requst for these kinds of statistical functions to esmlab?

# Example from http://xarray.pydata.org/en/stable/dask.html

def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

def pearson_correlation(x, y, dim):
    return xr.apply_ufunc(
        pearson_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

# Weighted coorelation
def covariance_gufunc_wgt(x, y, w):
    return (w * (x - (x*w).mean(axis=-1, keepdims=True)) *
            (y - (y*w).mean(axis=-1, keepdims=True))).mean(axis=-1)

def pearson_correlation_gufunc_wgt(x, y, w):
    return covariance_gufunc_wgt(x, y, w) / np.sqrt(
        covariance_gufunc_wgt(x, x, w) * covariance_gufunc_wgt(y,y,w))

def pearson_correlation_wgt(x, y, w, dim):
    return xr.apply_ufunc(
        pearson_correlation_gufunc_wgt, x, y, w,
        input_core_dims=[[dim], [dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

# rank correlation
def spearman_correlation_gufunc(x, y):
    x_ranks = bottleneck.rankdata(x, axis=-1)
    y_ranks = bottleneck.rankdata(y, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)

def spearman_correlation(x, y, dim):
    return xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

