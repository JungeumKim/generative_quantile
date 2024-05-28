import numpy as np
import torch
from IPython.core.debugger import set_trace


def X_forward_sampler(thetas, n=2, seed = 12345,np_random = None,
                      device="cuda",as_torch = False):
    if np_random is None:
        np_random = np.random.RandomState(seed)

    m = thetas.shape[0]
    x =  np_random.normal(thetas[:,0].reshape(-1,1),
                          thetas[:,1].reshape(-1,1)**(0.5),
                          size=(m,n))
    if as_torch:
        x = torch.from_numpy(x).float().to(device)
    return x

def tensor_forward_sampler(n = 2, theta_batch_size=100,x_batch_size= 100,
                           device="cuda",seed = 12345,
                            h_param={"nu":25, "sigma0_sq":1, "mu0":0,"kappa":1},
                            as_torch = False,
                            np_random = None):
    '''
    :return: (theta_batch_size x n x x_batch_size)
    '''
    if np_random is None:
        np_random = np.random.RandomState(seed)

    sigma_sq =h_param["nu"] * h_param["sigma0_sq"] / np_random.chisquare(h_param["nu"], size=theta_batch_size)
    theta = np_random.normal(h_param["mu0"], np.sqrt(sigma_sq/h_param["kappa"]), size=theta_batch_size)

    x =  np_random.normal(theta.reshape(theta_batch_size,1,1),
                          sigma_sq.reshape(theta_batch_size,1,1)**(0.5),
                          size=(theta_batch_size,n,x_batch_size))
    if as_torch:
        x = torch.from_numpy(x.reshape(theta_batch_size,n,x_batch_size)).float().to(device)
    return x

def forward_sampler(n = 2, batch_size=100,device="cuda",seed = 12345,
                    h_param={"nu":25, "sigma0_sq":1, "mu0":0,"kappa":1},
                    as_torch = False,
                    np_random = None):

    if np_random is None:
        np_random = np.random.RandomState(seed)

    sigma_sq =h_param["nu"] * h_param["sigma0_sq"] / np_random.chisquare(h_param["nu"], size=batch_size)
    theta = np_random.normal(h_param["mu0"], np.sqrt(sigma_sq/h_param["kappa"]), size=batch_size)
    
    parameters = np.column_stack((theta,sigma_sq))

    x =  np_random.normal(theta.reshape(batch_size,1),
                          sigma_sq.reshape(batch_size,1)**(0.5),
                          size=(batch_size,n))
    if as_torch:
        x = torch.from_numpy(x.reshape(-1,n)).float().to(device)
        parameters = torch.from_numpy(parameters).float().to(device)
    return parameters, x

def gauss_gen(parameters, as_torch=False,np_random = None,
              n = 2, batch_size=100,device="cuda",seed = 12345):
    
    if np_random is None:
        np_random = np.random.RandomState(seed)
        
    theta = parameters[:,0]
    sigma_sq = parameters[:,1]
    x =  np_random.normal(theta.reshape(batch_size,1),
                          sigma_sq.reshape(batch_size,1)**(0.5),
                          size=(batch_size,n))
    if as_torch:
        x = torch.from_numpy(x.reshape(-1,n)).float().to(device)
        parameters = torch.from_numpy(parameters).float().to(device)
    return x



def posterior_sampler(X = 2, batch_size=100,device="cuda",seed = 12345,
                     h_param={"nu":25, "sigma0_sq":1, "mu0":0,"kappa":1},
                     as_torch = False,
                     np_random = None):
    '''
    X: an (nx1) matrix.
    '''
    if np_random is None:
        np_random = np.random.RandomState(seed)

    n = len(X)
    bar_x = X.mean().item()
    kappa_n = n+h_param["kappa"]
    nu_n = n+ h_param["nu"]
    s2 = ((X-bar_x)**2).sum().item()/(n-1)

    mu_n = (n*bar_x + h_param["kappa"]*h_param["mu0"])/(n+h_param["kappa"])
    nu_n_sigma_sq_n = h_param["nu"] * h_param["sigma0_sq"] + (n-1)*s2 + (bar_x-mu_n)**2/(1/h_param["kappa"]+1/n)

    sigma_sq = nu_n_sigma_sq_n / np_random.chisquare(nu_n, size=batch_size)
    theta = np_random.normal(mu_n, np.sqrt(sigma_sq/kappa_n), size=batch_size)
    
    if as_torch:
        theta = torch.from_numpy(theta).float().to(device)
        sigma_sq = torch.from_numpy(sigma_sq).float().to(device)
    return theta, sigma_sq



