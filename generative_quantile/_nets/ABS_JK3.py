"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), implemented by Jungeum Kim.

TODO) the data should be normalized if want to mimic the original code by Wuexi Wang.
TODO) clean up device

TODO) JK2: use all my network designs and learning schedule for training. Adam update principle. and, my learning rate.
"""

from IPython.core.debugger import set_trace
from _nets.basic_nets import  MLP
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time

from _nets.brenier_maps import DeepSets,BiRNN

import matplotlib.pyplot as plt
import seaborn as sns
from _utils.distances import compute_mmd,compute_dtm
import pandas as pd



class Auto_ss(nn.Module):
    def __init__(self,  f1_dim=2, f2_dim=2, device="cuda", x_dim=2,
                 factor=64, f1_layers =3,*args, **kwargs):
        super().__init__()
        self.f1_dim = f1_dim
        self.f2_dim= f2_dim
        if  self.f1_dim>0:
            self.f1 = DeepSets(dim_x=x_dim,
                              dim_ss=self.f1_dim,
                              factor=factor, num_layers=f1_layers, device=device)
        if  self.f2_dim>0:
            self.f2 = BiRNN(input_size=x_dim,
                   hidden_size=512,
                   num_layers=1,
                   xdim=self.f2_dim)

        self.device=device
        self.to(device)

    def forward(self, x):
        if self.f1_dim>0 and  self.f2_dim ==0:
            return self.f1(x)
        elif self.f1_dim==0 and  self.f2_dim >0:
            return self.f2(x)
        else:
            x= torch.cat([self.f1(x),self.f2(x)], 1)
            return x



class Generator(nn.Module):

    def __init__(self, x_dim=2,x_length=1, theta_dim = 2,  dropout = 0.1,
                  activation = "relu", lower_bound=None, f_manual = None, ss_f = False, f1_dim=2,f2_dim=2
                 ,d_hidden = [128,128,128]):
        super().__init__()
        self.ss_f = ss_f
        self.d_cond = f1_dim + f2_dim
        
            
        if f_manual is not None: 
            print("manual feature map")
            self.f = f_manual
            
        elif ss_f:
            print("deepset feature map")
            self.f = Auto_ss(f1_dim=f1_dim, f2_dim=f2_dim, 
                             x_dim=x_dim, theta_dim = theta_dim)
        else:
            print("mlp feature map")
            self.f = MLP(dim=x_dim*x_length, z_dim=self.d_cond)

        self.d_noise = theta_dim
        self.theta_dim =theta_dim
        
        d_in = [self.d_noise + self.d_cond] +d_hidden
        d_out = d_hidden + [theta_dim]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.leaky_relu
        self.lower_bound=lower_bound

    def forward(self, context, noise = None):
        # context: conditioning variable.
        if noise is None:
            noise = torch.randn(context.size(0), self.d_noise).to(context.device)
        
        f_context = self.f(context)
        
        x = torch.cat([noise, f_context], -1)

        for layer in self.layers:
                x = self.dropout(self.activation(layer(x)))
        
        if self.lower_bound is not None:
            # it is only for the special case where the second element is known to be lowerbounded e.g., by 0.
            x[:,1]  = x[:,1].clip(min=self.lower_bound)
        return x


class Critic(nn.Module):

    def __init__(self,activation = "relu",dropout = 0,input_dim=2, d_cond = 4, d_hidden = [128,128,128]):

        super().__init__()
        self.d_cond = d_cond
        
        d_in = [input_dim+d_cond] + d_hidden
        d_out = d_hidden + [1]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.leaky_relu

    def forward(self, x, context):
        
        x = torch.cat([x, context], -1)
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        
        return self.layers[-1](x)

    def gradient_penalty(self, x, x_hat, context):

        alpha = torch.rand(x.size(0)).unsqueeze(1).to(x.device)
        interpolated = x * alpha + x_hat * (1 - alpha)
        interpolated = torch.autograd.Variable(interpolated.detach(), requires_grad=True)
        critic = self(interpolated, context)
        gradients = torch.autograd.grad(critic, interpolated, torch.ones_like(critic),
                                        retain_graph=True, create_graph=True, only_inputs=True)[0]
        penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()             # one-sided
        # penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()          # two-sided
        return penalty


class ABS():

    def __init__(self, simulator, theta_dim,  x_dim, 
                 x_length,parallel=False,
                 f_manual=False, ss_f = False, f1_dim=2, f2_dim=5, 
                 device="cuda",epoch=150, batch_size = 200, 
                 seed=1234, lower_bound=None,
                 n_iter=100,
                 vis_every = 20,observed_data = None, 
                 true_post = None,
                 true_params=None, Xs=None,posterior_sampler =None,
                 *args, **kwargs):
        
        self.ss_f = ss_f

        self.generator = Generator(
                                   x_dim,x_length,
                                   theta_dim = theta_dim,
                                   dropout = 0.1,
                                   activation = "relu",
                                   lower_bound=lower_bound, 
                                   
                                   f_manual=f_manual, ss_f = ss_f,f1_dim=f1_dim, f2_dim=f2_dim,
                                    d_hidden = [128,128,128])
        
        self.critic = Critic(activation = "relu",
                             dropout = 0,
                             input_dim=theta_dim,
                             d_cond = f1_dim+ f2_dim,
                             d_hidden = [128,128,128])

        self.np_random = np.random.RandomState(seed)
        self.generator.to(device), self.critic.to(device)
        self.simulator = simulator
        self.device = device
        self.epoch = epoch
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.x_dim=x_dim
        self.x_length=x_length
        
        
        
        if parallel and (torch.cuda.device_count() > 1):
            self.critic = torch.nn.DataParallel(self.critic)
            self.generator = torch.nn.DataParallel(self.generator)
            print("data parallel")
        self.critic.to(device)
        self.generator.to(device)
        
        self.vis_every= vis_every
        self.observed_data =observed_data
        self.true_post = true_post
        self.true_params=true_params
        self.Xs=Xs
        self.posterior_sampler = posterior_sampler
        if (self.true_params is not None) and (self.Xs is not None) and (self.posterior_sampler is not None):
            self.mmd_measuring=True
        else:
            self.mmd_measuring=False

        if true_post is None:
            self.vis_every = 999999
        self.current_epoch = 0
        
        
    def train(self, critic_gp_factor = 5,
              critic_lr = 0.01,
              critic_steps = 15,
              generator_lr = 0.01,
              print_every=20,
              test_iter=10):
        generator = self.generator
        critic = self.critic
        simulator = self.simulator
        # setup training objects
        start_time = time()
        local_start_time = time()
        step = 1
        n_iter = self.n_iter * critic_steps

        self.log = []
            
        for epoch in range(1,self.epoch+1):
            running_loss = 0.0
            
            print(f"Epoch {epoch}")
            opt_generator = optim.Adam(generator.parameters(), lr=generator_lr*(0.99**epoch))
            opt_critic = optim.Adam(critic.parameters(), lr=critic_lr*(0.99**epoch))

            # train loop
            WD_train, WD_test= 0, 0
            n_critic = 0
            critic_update = True

            for iter in range(n_iter):
                x, context = simulator(batch_size = self.batch_size,np_random = self.np_random)
                x, context = x.to(self.device), context.to(self.device)
                if not self.ss_f:
                    context = context.view(-1,self.x_dim*self.x_length)
                generator.zero_grad()
                critic.zero_grad()
                
                x_hat = generator(context)
                g_context = generator.f(context)
                
                critic_x_hat = critic(x_hat, g_context).mean()


                if n_critic < critic_steps:
                    critic_x = critic(x, g_context).mean()
                    WD = critic_x - critic_x_hat
                    loss = - WD
                    loss += critic_gp_factor * critic.gradient_penalty(x, x_hat, g_context)
                    loss.backward()
                    opt_critic.step()
                    WD_train += WD.item()
                    n_critic += 1

                else: #generator 1 step.
                    loss = - critic_x_hat
                    loss.backward()
                    opt_generator.step()
                    running_loss += loss.item()
                    n_critic = 0 # now, the critic will again be trained.

                step += 1
            WD_train /= n_iter
            # test loop

            loss_cum = running_loss/self.n_iter
            sample = self.sampler(self.observed_data,300,shaper=lambda x: x)
            mmd = compute_mmd(sample, self.true_post) if self.true_post is not None else 0
            (Emmd,Edtm) = self.mmd_val() if self.mmd_measuring else (0,0)
            self.log.append({"loss":loss_cum, "mmd": mmd, "Emmd":Emmd, "Edtm":Edtm})
            
            if epoch % self.vis_every ==0:
                try:
                    self.vis(sample)
                except:
                    print("some vis err")

    def sampler(self, X, sample_size,shaper = None):
        if shaper is None:
            X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1)
        else: 
            X = shaper(X)
            
        X = X.to(self.device)
        with torch.no_grad():
            if not self.ss_f:
                X = X.view(-1,self.x_dim*self.x_length)
            return self.generator(X).to("cpu")

    def mmd_val(self):
        mmds = []
        dtms = []
        for i in range(self.Xs.shape[0]):
            true_param, observed_data = self.true_params[i],self.Xs[i]
            sim_post_sample = self.sampler(observed_data,300)
            theta, sigma_sq = self.posterior_sampler(X = observed_data)
            true_post_sample = np.stack([theta,sigma_sq],1)
            mmd_value = compute_mmd(sim_post_sample,true_post_sample)
            dtm_value = (((true_param-sim_post_sample.numpy())**2).sum(1)**0.5).mean()
            mmds.append(mmd_value)
            dtms.append(dtm_value)
        return np.mean(mmds),np.mean(dtms)
    
    def vis(self,sample):
        df = pd.DataFrame(self.log)
        n_col = len(df.keys())+1
        
        fig,axis = plt.subplots(1,n_col, figsize=(4*n_col,4))#, sharex=True, sharey=True)
        
        ax = axis[0]
        ax.set_title(f"Epoch {self.current_epoch}")
        sns.kdeplot(x=sample[:,0], y=sample[:,1], ax=ax, fill=False)
        sns.kdeplot(x=self.true_post[:,0], y=self.true_post[:,1], ax=ax, fill=True)
        
        for i, key in enumerate(df.keys()):
            
            ax = axis[1+i]
            df[key].plot(ax = ax)
            ax.set_title(key)
        
        plt.show()

    def save(self, path):
        if isinstance(self.generator, torch.nn.DataParallel):
            torch.save({
            'generator': self.generator.module.state_dict(),
            'critic': self.critic.module.state_dict()
        }, path)
            
        else:
            torch.save({
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict()
            }, path)

    def load(self, path):
        saved = torch.load(path)
        self.generator.load_state_dict(saved['generator'])
        self.critic.load_state_dict(saved['critic'])

