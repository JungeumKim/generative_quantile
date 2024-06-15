import numpy as np
from scipy.spatial.distance import pdist, squareform

def rbf_kernel(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    pairwise_sq_dists = pdist(np.vstack([X, Y]), 'sqeuclidean')
    K = np.exp(-gamma * squareform(pairwise_sq_dists))
    return K[:X.shape[0], X.shape[0]:]
def compute_mmd(X, Y, kernel=rbf_kernel):
    m = X.shape[0]
    n = Y.shape[0]
    
    # Kernel computations
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)
    
    # MMD computation
    mmd = np.sum(K_XX) / (m * m) + np.sum(K_YY) / (n * n) - 2 * np.sum(K_XY) / (m * n)
    return np.sqrt(mmd)



def compute_dtm(X, samples):
    
    dtm_value = (((X-samples)**2).sum(1)**0.5).mean()
    
    return dtm_value

def distance_giver(sampler, posterior_sampler, true_params, Xs, n_test=300):
    mmds = []
    dtms = []
    for i in range(100):
        true_param, observed_data = true_params[i],Xs[i]
        sim_post_sample = sampler(observed_data,n_test)
        theta, sigma_sq = posterior_sampler(observed_data,n_test)

        true_post_sample = np.stack([theta,sigma_sq],1)
        mmd_value = compute_mmd(sim_post_sample,true_post_sample)
        mmds.append(mmd_value)
        
        dtm_value = (((true_param-sim_post_sample.numpy())**2).sum(1)**0.5).mean()
        dtms.append(dtm_value)
        
    return {"mmd":mmds,"dtm":dtms}