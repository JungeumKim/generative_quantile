from IPython.core.debugger import set_trace
import numpy as np
from scipy.stats import invwishart
import torch

def forward_sampler(batch_size=100,device="cuda",seed = 12345,
                    h_param={"m":10, "mu0": np.array([0,0,0,0]),
                             "nu":4, "Psi_0": np.diag([1, 1, 1, 1])},
                    as_torch = False,
                    np_random = None):

    if np_random is None:
        np_random = np.random.RandomState(seed)

    Sigma = invwishart.rvs(h_param["nu"], h_param["Psi_0"], size=batch_size, random_state=np_random) # batch_size x d x d
    mus = []
    for sig in Sigma: # sig: d x d, Sigma: n x d x d
        mus.append(np_random.multivariate_normal(h_param["mu0"], sig/h_param["m"]))
    mu = np.expand_dims(np.stack(mus),2) # n x d x 1

    parameters = np.concatenate((mu, Sigma), axis=2) # n x d x (1+d)

    samples = []
    for i in range(batch_size):
        samples.append(np.random.multivariate_normal(mus[i], Sigma[i]))
    x = np.stack(samples)

    if as_torch:
        x = torch.from_numpy(x).float().to(device)
        parameters = torch.from_numpy(parameters).float().to(device)

    return parameters, x

def posterior_sampler(X = 2, batch_size=100,device="cuda",seed = 12345,
                     h_param={"m":10, "mu0": np.array([0,0,0,0]),
                             "nu":4, "Psi_0": np.diag([1, 1, 1, 1])},
                     as_torch = False,
                     np_random = None):
    '''
    X: an (nxd) matrix.
    '''
    if np_random is None:
        np_random = np.random.RandomState(seed)
    (n,d) = X.shape
    m, nu, Psi_0, mu0 = h_param["m"], h_param["nu"], h_param["Psi_0"], h_param["mu0"]
    xbar = X.mean(0)
    X_centered = X - xbar
    S = np.dot(X_centered.T, X_centered)/n
    Sigma_post = invwishart.rvs(nu+n, Psi_0 + n*S + \
                            m*n/(n+m)*np.outer((xbar-mu0), (xbar-mu0)),
                            size=batch_size)
    center_post = (n*xbar + m*mu0)/(n+m)
    mus = []
    for sig in Sigma_post: # sig: d x d, Sigma: n x d x d
        mus.append(np_random.multivariate_normal(center_post, sig/(n+m)))
    mu_post = np.expand_dims(np.stack(mus),2) # n x d x 1

    if as_torch:
        mu_post= torch.from_numpy(mu_post).float().to(device)
        Sigma_post= torch.from_numpy(Sigma_post).float().to(device)

    return mu_post, Sigma_post
