import numpy as np
from scipy.stats import norm,invgamma
from IPython.core.debugger import set_trace

class Conjugate_Model:
    def __init__(self, 
                 input_dim=2,
                 seed=59,
                 eps_discount_factor = 0.9,n_repeats = 1000, initial_eps = 2.0,
                 h_param={"nu":25, "sigma0_sq":1, "mu0":0,"kappa":1},
                 obs_data=np.array([0,0]), support = np.array([[-10,10],[0,10]])):

        self.support = support
        self.observed_data = obs_data
        self.epsilon_schedule = initial_eps * eps_discount_factor ** np.arange(n_repeats)
        self.bandwidth = 0.5
        self.input_dim =input_dim
        self.np_random = np.random.RandomState(seed)
        self.h_param = h_param

    def prior_sim(self, size):
        h_param = self.h_param 
        sigma_sq =h_param["nu"] * h_param["sigma0_sq"] / self.np_random.chisquare(h_param["nu"], size=size)
        theta = self.np_random.normal(h_param["mu0"], np.sqrt(sigma_sq/h_param["kappa"]), size=size)
        parameters = np.column_stack((theta,sigma_sq))
        return parameters

    def prior_density(self, thetas):
        h_param = self.h_param 
        p_gauss = norm.pdf(thetas[:,0], loc=h_param["mu0"], scale=np.sqrt(thetas[:,1]))
        p_inverse_gamma = invgamma.pdf(thetas[:,1], a=h_param["nu"]/2, 
                                       scale=h_param["sigma0_sq"]*h_param["nu"]/2)
        return p_gauss*p_inverse_gamma

    def simulator(self, thetas, noise_std=0):
        '''
        Simulates data from the parameter
        thetas: n x 2 : cetner and scale.
        '''
        batch_size = thetas.shape[0]
        x =  self.np_random.normal(thetas[:,0].reshape(-1,1),
                          thetas[:,1].reshape(-1,1)**(0.5),
                          size=(batch_size,self.input_dim))
        return x

    def discrepancy(self, thetas, xobs):
        sims = self.simulator(thetas)
        masked_gap = (sims.reshape(-1,self.input_dim) - xobs.reshape(1,self.input_dim))
        d = np.sqrt(np.sum(masked_gap**2, axis=1))
        return d

