import numpy as np

def integrate(sigma_nu, alpha, delta,
              np_random=None, seed=1234, n_steps=200):
    
    if np_random is None:
        np_random = np.random.RandomState(seed)
    
    batch_size = len(sigma_nu)
    
    log_h_prev = np.zeros(batch_size)
    
    log_h=[] 
    log_y_sq=[] 
    
    for _ in range(n_steps):
        log_h_t = alpha + delta * log_h_prev + sigma_nu * np_random.normal(size = batch_size)
        log_y_sq_t = log_h_t + np.log(np_random.normal(size = batch_size)**2)
        
        log_h.append(log_h_t)
        log_y_sq.append(log_y_sq_t)
        
        log_h_prev = log_h_t
        
    return np.column_stack(log_y_sq),np.column_stack(log_h)

def simulate(batch_size = 100, np_random=None, seed=1234,
             n_steps=200, h_params = {"nu0": 10, "s0": 1, "beta" : [0,0],"sigma": 1}):
    if np_random is None:
        np_random = np.random.RandomState(seed)
        
    sigma_nu = 1/np_random.gamma(shape = h_params["nu0"],scale =h_params["s0"],
                           size=batch_size)
    alpha = np_random.normal(loc = h_params["beta"][0], 
                             scale = sigma_nu*h_params["sigma"], 
                             size=batch_size)
    delta = np_random.normal(loc= h_params["beta"][1], 
                              scale = sigma_nu*h_params["sigma"],
                              size=batch_size)
        
    log_y_sq,log_h = integrate(sigma_nu = sigma_nu, alpha = alpha, delta = delta, 
                    np_random=np_random, 
                    n_steps=n_steps)
    
    thetas = np.column_stack([sigma_nu, alpha, delta])
    
    return thetas, log_y_sq,log_h