from IPython.core.debugger import set_trace
import numpy as np
import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import torch
from os.path import join
import pathlib
import sys
import argparse
import pathlib

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import scipy

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def unif(size, eps=1E-7, device ="cuda"):
    return torch.clamp(torch.rand(size).to(device), min=eps, max=1-eps)

def parse_one():
    #Arguments:
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--exp_dir', default ="../configs/nat0.json")
    exp_dir, lest = parser.parse_known_args()
    return exp_dir, lest


def main(args):
    
    print("Input arguments:")
    for key, val in vars(args).items(): print("{:16} {}".format(key, val))

    ID_ = F"gauss{args.gauss}_efficient{args.efficient}"
    net_path = join(args.exp_dir,F"results/trained_nets/{ID_}")
    vis_path = join(args.exp_dir,F"results/visualizations/{ID_}")
    pathlib.Path(net_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(vis_path).mkdir(parents=True, exist_ok=True)
    
    HPARAM = {"nu":args.nu, "sigma0_sq":args.sigma0_sq, 
              "mu0":args.mu0,"kappa":args.kappa}
    np_random = np.random.RandomState(args.seed)
    
    
    
    net = ConditionalConvexQuantile(xdim=args.n_sample,
                                ydim=2, 
                                a_hid=512,
                                a_layers=3,
                                b_hid=512,
                                b_layers=3)

    net.to(args.device)
    
    gauss = torch.distributions.normal.Normal(torch.tensor([0.]).to(args.device), 
                                              torch.tensor([1.]).to(args.device))
    for epoch in range(1, args.epoch+1):
        optimizer = optim.Adam(net.parameters(), lr=args.lr*(0.99**epoch))
        running_loss = 0.0
        for idx in range(args.n_iter):
        #for idx, (Y, label) in enumerate(loader):
            Y, label = forward_sampler(n = args.n_sample, 
                                batch_size=args.batch_size,
                                device=args.device,
                                h_param=HPARAM,
                                as_torch = True,
                                np_random = np_random)
            u = unif(size=(args.batch_size, args.n_sample), 
                     device=args.device)
            if args.gauss:
                u = gauss.icdf(u)
            optimizer.zero_grad()
            X,_ = net.f(label.unsqueeze(1))
            alpha, beta= net(u)
            loss = dual_JK(U=u, Y_hat=(alpha, beta), 
                           Y=Y, X=X, eps=0, efficient=args.efficient)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print('%.5f' %(running_loss))
        
        #if epoch % 100 == 0:
            #print('%.5f' %(running_loss))#/(idx+1)))
            #test(net, args.n_sample, args.gauss, HPARAM, 
            #     test_val=2.2, n_test = 500, 
            #     save_dir=vis_path+F"/epoch{epoch}.png")
            #net.train()

    
    torch.save(net.state_dict(),  net_path+F"/net.net")
    print(F"Network has been saved at {net_path}/net.net")
if __name__ == '__main__':

    args_one, args_lest = parse_one()
    sys.path.append(args_one.exp_dir)

    from config import parse_args

    args = parse_args(args_lest, name_space=args_one)

    if hasattr(args, "worktree"):
        sys.path.insert(0, args.worktree) 
    else:
        sys.path.insert(0, "/home/kim2712/Desktop/research/generative_quantile/generative_qunatile")
               
    from brenier.models import ConditionalConvexQuantile,dual_JK
    from _utils.toy_2d_util import test
    #from _utils.breiner_util import plot2d, histogram, plotaxis
    from _data.gaussian_conjugate import forward_sampler#, posterior_sampler


    
    main(args)
