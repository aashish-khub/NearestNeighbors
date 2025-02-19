import numpy as np
import scipy as sp
import math as math
import random as rand
from scipy.stats import multivariate_normal

from .dnn import DistNNEstimator

def obs_Overlap(i, j, Masking) :
    """ Overlapping column indices for ith and jth row
    Input: 
      i, j: Row index i and j
      Masking: Masking matrix 
 
    Output:
      Vector with val 1 if overlapping and val 0 otherwise
    """
    T = Masking.shape[1]
    overlap = []
    for t in range(T) : 
        if Masking[i, t] == 1 and Masking[j, t] == 1:
            overlap.append(t)
            
    return(overlap)

def sqmmd_est2(dat1, dat2, kernel) :
    """ Computes U-statistics estimate of squared MMD_k, when number of samples from each distribution are different

    Input
        dat 1 : (m * d) data matrix coming from first d dim distribution
        dat 2 : (n * d) data matrix coming from second d dim distribution
        kernel : k(x, y) that defines MMD_k^2
    
    Output
        MMD_k^2 estimator
    """
    m = dat1.shape[0]
    n = dat2.shape[0]

    if dat1.shape[1] != dat2.shape[1] : 
        print("Data dimension do not match!")
        return
    
    d = dat1.shape[1]

    XX = np.matmul(dat1, np.transpose(dat1)) # m by m matrix with x_i^Tx_j
    YY = np.matmul(dat2, np.transpose(dat2)) # n by n matrix with y_i^Ty_j
    XY = np.matmul(dat1, np.transpose(dat2)) # m by n matrix with x_i^Ty_j  

    if kernel == "linear" :
        kXX, kYY, kXY = XX, YY, XY
    if kernel == "square" :
        kXX, kYY, kXY = (XX + np.ones( (m, m) ))**2, (YY + np.ones( (n, n) ))**2, (XY + np.ones( (m, n) ))**2
    if kernel == "exponential" :
        dXX_mm = np.vstack((np.diag(XX), )*m) # m*m matrix : each row is the diagonal x_i^Tx_i
        dYY_nn = np.vstack((np.diag(YY), )*n) # n*n matrix : each row is the diagonal y_i^Ty_i
        dXX_mn = np.vstack((np.diag(XX), )*n).transpose() # m*n matrix : each row is the diagonal x_i^Tx_i
        dYY_mn = np.vstack((np.diag(YY), )*m) # m*n matrix : each row is the diagonal y_i^Ty_i

        kXX = np.exp( -0.5*( dXX_mm + dXX_mm.transpose() - 2*XX ) ) 
        kYY = np.exp( -0.5*( dYY_nn + dYY_nn.transpose() - 2*YY ) )
        kXY = np.exp( -0.5*( dXX_mn + dYY_mn - 2*XY ) )
        
    val = (kXX.sum() - np.diag(kXX).sum())/(m*(m - 1)) + (kYY.sum() - np.diag(kYY).sum())/(n*(n - 1)) - 2*kXY.sum()/(n*m)
    if val < 0 : 
        val = 0

    return(val)

def row_Metric(i, j, t, Data, Masking, kernel, exc_opt) : 
    """ Computes MMD_k^2 based metric between rows of Distributional Matrix~(DM) with missingness

    Input
        i, j : two row indices under comparison
        t : column of interest - so need to exclude this column
        kernel : baseline kernel
        Data : (N * T * n * d) array 
        Masking : (N * T) sized matrix indicating which entries are observed(val = 1)
        exc_opt : if True, omit t th column when constructing row metric, if False, include t th column
        
    Output
        Metric between row i, j when the target parameter is on the t th column
    """

    N = Masking.shape[0]
    T = Masking.shape[1]
    
    overlap = obs_Overlap(i, j, Masking)

    if exc_opt == True : 
        if sum(np.isin(overlap, t)) == 1 : 
            overlap = np.delete(overlap, np.where(overlap == t)[0])
    if len(overlap) == 0 : 
        val = 10**5
        return(val)

    Data_i = Data[i, :, :, :]
    Data_j = Data[j, :, :, :]

    pre_val = np.zeros(len(overlap))
    for tau in range(len(overlap)) :
        tau_ind = overlap[tau]
        pre_val[tau] = sqmmd_est2(Data_i[tau_ind, :, :], Data_j[tau_ind, :, :], kernel)
    
    val = sum(pre_val)/len(overlap)

    return(val)

def row_mmDNN(i, t, Data, row_Dissim_vec, Masking, eta) : 
    """ Implements DNN with MMD_k^2 to impute (i, t) entry using eta radius
    (Note that kernel information is already encoded in row_Dissim_vec)
    (This ftn regards the (i) identification of neighbor and (ii) averaging)

    t th column of Data is used for averaging
    t th column of Masking is used to pick the ones observed ... (1)
    row_Dissim_vec, eta are used to pick the ones within neighborhood ... (2)
    when intersecting (1) and (2), make sure to exclude i th row and then take the barycenter

    Input
        i, t : index of target distribution - mu_{i, t}
        row_Dissim_vec : row-wise metric (rho_{i, j}) for j in [N], excluding t th column for construction
        Data : (N * T * n * d) array
        Masking : (N * T) array
        kernel : baseline kernel
        eta : radius
        
    Output
        (|neighbor| * n * d) sized array of all the neighboring measurements
    """
    N, T, n, d = Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3]

    neighbor_candidate = Data[:, t, :, :]

    neighbor_ind = np.where( (row_Dissim_vec < eta)*(Masking[:, t]) == 1 )[0]

    if sum(np.isin(neighbor_ind, i)) == 1 : # Pretending as IF (i, t) entry is missing
        neighbor_ind = np.delete(neighbor_ind, np.where(neighbor_ind == i)[0])

    if len(neighbor_ind) == 0 :
        neighbor = np.zeros( (n, d) )
        return(neighbor)

    neighbor = neighbor_candidate[neighbor_ind, :, :]

    neighbor = neighbor.reshape(-1, neighbor.shape[-1]) # ((|neighbor_ind| x n) * d) array

    return(neighbor)


def mmDNN_cv(Data, Masking, kernel, eta_cand) : 
    """ 2-fold CV error

    Input 
        Data : Full data that is split into two - train & test
        Masking : Full Masking matrix that is split into two - train & test
        eta_cand : Candidate of radius that is explored

    Output
        Optimal radius eta^star
    """
    N, T, n, d = Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3]

    if (T % 2) == 0 : 
        T_1 = int(T/2)
    if (T % 2) == 1 :
        T_1 = int((T + 1)/2)

    Data1 = Data[:, np.arange(T_1), :, :]
    Masking1 = Masking[:, np.arange(T_1)]

    Data2 = Data[:, np.arange(T_1, T), :, :]
    Masking2 = Masking[:, np.arange(T_1, T)]


    # Construct rho_{i, j} for rows using Data 1 ... (1)
    row_Dissim_mat = np.zeros( (N, N) )

    t_0 = 1 # Arbitrary column index when constructing rho_{i, j} under CV process
    for i in range(N - 1) :
        for j in range((i + 1), N) :
            row_Dissim_mat[i, j] = row_Metric(i, j, t_0, Data1, Masking1, kernel, exc_opt = False)
    
    row_Dissim_mat = row_Dissim_mat + np.transpose(row_Dissim_mat)
    
    # Estimate `observed` entries in Data 2 using rho_{i, j} in (1) ... (2)
    T_2 = T - T_1 # Length of second partition of Data

    perf = np.zeros(len(eta_cand))
    for eta_ind in range( len(eta_cand) ) :
        eta = eta_cand[eta_ind]
        mmd_error = np.zeros( (N, T_2) )
        for i in range(N) : 
            for t in range(T_2) :
                if Masking2[i, t] == 1 :
                    hat_mu_it = row_mmDNN(i, t, Data2, row_Dissim_mat[i, :], Masking2, eta)
                    mmd_error_it = sqmmd_est2(hat_mu_it, Data2[i, t, :, :], kernel)
                    mmd_error[i, t] = mmd_error_it
        perf[eta_ind] = mmd_error.sum()/(Masking2.sum())

    eta_star = eta_cand[np.argmin(perf)]
    
    return(eta_star)

def mmDNN_direct(i, t, Data, row_Dissim_vec, Masking, eta_cand, delta, kernel):
    """ A direct optimization over hyper-parameter eta, without using CV/Data-splitting
    
    Input 
        Data : Full data that is split into two - train & test
        Masking : Full Masking matrix that is split into two - train & test
        eta_cand : Candidate of radius that is explored

    Output
        Optimal radius eta^star
    """
    if kernel == "exponential":
        sup_kern = 1 
    if kernel == "square":
        sup_kern = 10 # Should change

    N, T, n, d = Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3]
    c_0 = 8*np.exp(1/(np.exp(1)))/np.sqrt(2*np.exp(1)*np.log(2))

    perf = []

    for m, eta in enumerate(eta_cand):
        neighborhood = np.where( (row_Dissim_vec < eta)*(Masking[:, t]) == 1 )[0] # Set of neighbors: (i) within eta distance (ii) observed

        if sum(np.isin(neighborhood, i)) == 1: # Pretending as if (i, t) entry is missing
            neighborhood = np.delete(neighborhood, np.where(neighborhood == i)[0])

        if len(neighborhood) == 0: # Default (null) output when there is zero neighbor
            perf.append(10**5) # Avoid selecting such eta without neighbors
        else:
            overlap = []
            for neighbor in neighborhood:
                overlap.append(np.sum(Masking[i, :]*Masking[neighbor, :]))

            Bias = 8*np.exp(1/np.exp(1))*sup_kern*np.log(2*N/delta)/(np.sqrt(2*np.log(2)*np.min(overlap)))
            Variance = 4*sup_kern*(np.log(n) + 1.5)/(n*len(neighborhood))

            perf.append(eta + Bias + Variance)

    eta_star, perf_star = eta_cand[np.argmin(perf)], perf[np.argmin(perf)]
    
    return(eta_star)

class DNNkernel(DistNNEstimator):
    pass
