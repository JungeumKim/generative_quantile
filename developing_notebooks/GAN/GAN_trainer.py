from IPython.core.debugger import set_trace

import io 
import os
import requests
import numpy as np
from numpy import expand_dims, mean, ones
from numpy.random import randn, randint
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import time
from sklearn.neighbors import KernelDensity
import pathlib
import sys
import argparse


def parse_one():
    #Arguments:
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--exp_dir', default ="../configs/nat0.json")
    exp_dir, lest = parser.parse_known_args()
    return exp_dir, lest

def main(args):

    device="cuda"

    nABC = args.n_train
    N_y=args.N_y #128
    p=N_y

    save_dir = F"{args.exp_dir}/nets/dataset{args.dataset}"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_checkpoint = save_dir + F"/M{args.Model}checkpoint_epoch{args.n_epoch}.ck"

    DS, HPARAM, _,_,_ = DU.setter(args.dataset)


    Xs,Ys, thetas = DS.sampler(N_y = N_y, batch_size = 2*nABC,
               h_param=HPARAM, device="cpu", seed = args.seed,
                              return_theta=True)
    #set_trace()
    
    if args.Model ==1:
        X, theta = Xs[:nABC], thetas[:nABC]
    else:
        X, theta = Xs[nABC:], thetas[nABC:]

    X_seq = X.float().numpy()
    theta_seq= theta.reshape(-1,1)

    theta_names=["theta"]
    X_names=["X"+str(i) for i in range(1,p+1)]

    df=pd.DataFrame(data=np.concatenate((theta_seq,X_seq),axis=-1),
                   columns=(theta_names+X_names))

    data_wrapper= wgan2.DataWrapper(df, continuous_vars=theta_names, context_vars=X_names)

    spec=wgan2.Specifications(data_wrapper, batch_size=args.batch_size, 
                              max_epochs=args.n_epoch, 
                              critic_lr=1e-3, generator_lr=1e-3,
                             print_every=10,device = device,
                              save_checkpoint = save_checkpoint,
                            save_every=10)

    generator=wgan2.Generator(spec)
    critic=wgan2.Critic(spec)

    thetas, Xs = data_wrapper.preprocess(df)

    wgan2.train(generator, critic, thetas, Xs, spec)


if __name__ == '__main__':
    
    args_one, args_lest = parse_one()
    sys.path.append(args_one.exp_dir)
    
    from config import parse_args
    
    args = parse_args(args_lest, name_space=args_one)
    
    if hasattr(args, "worktree"): 
        sys.path.insert(0, args.worktree)

    sys.path.insert(0, "/home/kim2712/Desktop/research/ai_selection/ai_selection")

    import ai_selection._data.neg_bin as DS
    from ai_selection._nets import wgan2
    import ai_selection._data.data_utils as DU

    main(args)

