## Modified from Gretchen's CLM_SVD notebook
## functions used in SVD used here 

import numpy as np

from scipy import stats


def decompose(ts_anomaly):
    ## assumes 2d anomalies, month*year
    ## Deconstruct the observations into U and V (matrices) and s (list)
    ## Outputs are vectors (dim = Nyear*Nmonths) & weights (dim = Nyear*Nyear)

    Nyears=ts_anomaly.shape[1]
    Nmonths=ts_anomaly.shape[0]
    # print(Nmonths, Nyears)

    if Nyears > Nmonths:
        U, s, V = np.linalg.svd(ts_anomaly, full_matrices=True)
    else:
        U, s, V = np.linalg.svd(ts_anomaly, full_matrices=False)

    ##Convert s from list to a diagonal matrix
    S = np.diag(s)

    # Initialize matrices to look at the first 2 (of nyears) singular vectors.
    # sv_vectors will represent the vector shapes
    # sv_weights will represent the annual weights for each vector
    # (2 singular vectors of interest, 12 months per year, n total years)
    sv_vectors = np.zeros((Nyears,Nmonths),dtype='float')
    sv_weights = np.zeros((Nyears,Nyears),dtype='float')

    # NOTE, this only works if Nyears > 12, otherwise, use S
    if Nyears > Nmonths:
        Sigma = np.zeros((Nmonths, Nyears))     # create m x n Sigma matrix
        Sigma[:Nmonths, :Nmonths] = np.diag(s)  # populate Sigma with n x n diagonal matrix

    ##Here we define the vector shapes by taking the dot product of U and S. 
    ##0 and 1 refer to the first and second singular vector, respectively
    for iyear in range(Nyears):
        if Nyears > Nmonths:
            sv_vectors[iyear,:]=np.dot(U,Sigma)[:,iyear]
            sv_weights[iyear,:]=V[iyear,:]  
            
        else:
            sv_vectors[iyear,:]=np.dot(U,S)[:,iyear]
            sv_weights[iyear,:]=V[iyear,:]      

    return(sv_vectors, sv_weights)

def calc_redistribution(sv_vectors, sv_weights, ts_anomaly):
    ##Calculate redistribution values for SV 1 and 2
    Nvec=sv_vectors.shape[0]
    Nmonth=int(ts_anomaly.shape[0])
    Nyear=int(ts_anomaly.shape[1])
    #print('Nvec, Nmon, Nyear are ' str(Nvec), Nmonth, Nyear)
    
    sv_theta = np.zeros(Nvec)
    for i in range(Nvec):
        sv_theta[i] = np.abs(np.nansum(sv_vectors[i,:]))/np.nansum(np.abs(sv_vectors[i,:]))
    
    # print(sv_theta)

    ##Calculate percentage variability described by sv vectors
    #First arrange timeseries (Obs and SV contributions)
    matrix_shape = np.shape(ts_anomaly) #first entry is months, second entry is years
    obs_timeseries = np.ravel(ts_anomaly)
  
    sv_matrix=np.zeros((Nvec, Nyear*Nmonth))
    for ivec in range(Nvec): #loop over vectors
        sv_timeseries=[]
        for iyear in range(Nyear): #loop over years
            sv_timeseries.append(sv_vectors[ivec,:]*sv_weights[ivec,iyear])
        sv_timeseries=np.ravel(np.transpose(sv_timeseries))
        sv_matrix[ivec,:]=sv_timeseries
        
      
    #Next calculate R^2 values
    sv_var_fraction = np.zeros(Nvec)
    for ivec in range(Nvec):
        sv_var_fraction[ivec] = stats.linregress(obs_timeseries,sv_matrix[ivec,:])[2]**2
         #sv_var_fraction[1] = stats.linregress(obs_timeseries,sv2_timeseries)[2]**2
         #sv_var_fraction[2] = stats.linregress(obs_timeseries,sv3_timeseries)[2]**2
    return(sv_theta, sv_var_fraction)



