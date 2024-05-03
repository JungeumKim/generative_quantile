import argparse
import numpy as np
import torch
import random


'''
        args: 
            -most important: working_dir, exp_path, exp_num
'''

def parse_args(args, name_space):
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--worktree',
                        default = "/home/kim2712/Desktop/research/generative_quantile/generative_qunatile",
                        help = "parent directory")
    parser.add_argument('--exp_dir', default = "./",  help= 'global path')
    parser.add_argument('--device', default = "cuda",  help= ' ')    
    
    parser.add_argument('--gauss', default = True)
    #parser.add_argument('--no-gauss', dest='gauss', action='store_false')
    #parser.set_defaults(gauss=False)
    
    parser.add_argument('--efficient', default = True)
    #parser.add_argument('--no-efficient', action='store_false')
    #parser.set_defaults(efficient=True)
    
    parser.add_argument('--epoch', type=int, default = 150, help= ' ')
    parser.add_argument('--n_iter', type=int, default = 100, help= ' ')
    parser.add_argument('--batch_size', type=int, default = 128, help= ' ')
    
    parser.add_argument('--lr', type=float, default = .01)
    parser.add_argument('--n_sample',  type=int, default = 2)
    
    parser.add_argument('--nu', type=int, default = 25, help= ' ')
    parser.add_argument('--sigma0_sq', type=float, default = 1)
    parser.add_argument('--mu0', type=float, default = 0)
    parser.add_argument('--kappa', type=int, default = 2, help= ' ')
    
    parser.add_argument('--seed', type=int, default = 12345, help= ' ')
    
    #learning setting:
    args = parser.parse_args(args, namespace=name_space) #parser.parse_args()
    return args

