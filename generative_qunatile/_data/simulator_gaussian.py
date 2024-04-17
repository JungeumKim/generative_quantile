import numpy as np
import rlabc
from IPython.core.debugger import set_trace
import torch
from scipy.stats import norm,invgamma
import sys
import os


#PATH = "/home/kim2712/Desktop/research/rlabc/"

class Conjugate_Model:
    def __init__(self, 
                 theta_true=np.array([-4.7956, -2.4520]),
                 input_dim=2,
                 #support=np.array([[-12.0, 12.0], [-12.0, 12.0]]), 
                 seed=59,
                 eps_discount_factor = 0.9,n_repeats = 1000, initial_eps = 2.0,
                 #path=PATH,
                 h_param={"nu":25, "sigma0_sq":1, "mu0":0,"kappa":1},
                 obs_data=np.array([0,0]), support = np.array([[-10,10],[0,10]])):
        



        #sys.path.insert(0,os.path.join(path, "developing_notebooks/GAN/"))#here absolute path to rlabc
        self.support = support
        self.theta_true = theta_true

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

    def d(self, sims, xobs): # We may need a different discrepancy function than this
        # Also, for masked data, we need to save the mask coordinates and the distance to simulated
        # data should only be computed on those masked coordinates, not on all coordinates.
        """Calculates distance between simulated data sims (n, 784) and xobs (1,784)"""

        masked_gap = (sims.reshape(-1,self.input_dim) - xobs.reshape(1,self.input_dim))

        return np.sqrt(np.sum(masked_gap**2, axis=1))

    def discrepancy(self, thetas, xobs):
        """Generates data at thetas (n,d) and then calculates distance
        to xobs (1,d) to result in discrepancy vector (n,)"""
        return self.d(self.simulator(thetas), xobs)

