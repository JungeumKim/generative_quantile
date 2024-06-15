from IPython.core.debugger import set_trace
from _nets.basic_nets import  MLP
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from _nets.brenier_maps import DeepSets,BiRNN

import matplotlib.pyplot as plt
import seaborn as sns
from _utils.distances import compute_mmd,compute_dtm
import pandas as pd


class AutoNet_manual(nn.Module):
    def __init__(self,  f_manual, f1_dim=2, f2_dim=2, device="cuda", x_dim=2, theta_dim = 2,batchnorm=True,
                 leaky=0.1, factor=64, n_layers=2, seed=1234,f1_layers =3,*args, **kwargs):
        super().__init__()
        
        self.batch = nn.BatchNorm1d(f1_dim+f2_dim)
        self.f_manual = f_manual
           
        self.auto_net = AutoNet(device="cuda", x_dim=f1_dim+f2_dim, theta_dim = theta_dim,
                                leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        self.device=device
        self.to(device)
        self.batchnorm = batchnorm
    def forward(self, X,taus):
        if self.batchnorm:
            fX = self.batch(self.f_manual(X))
        else:
            fX = self.f_manual(X)
        return self.auto_net(fX,taus)
    
class AutoNet_ss(nn.Module):
    def __init__(self,  f1_dim=2, f2_dim=2, device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234,f1_layers =3,*args, **kwargs):
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

        self.f = self.f_automatic
        self.auto_net = AutoNet(device="cuda", x_dim=f1_dim+f2_dim, theta_dim = theta_dim,
                                leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        self.device=device
        self.to(device)

    def f_automatic(self, x):
        if self.f1_dim>0 and  self.f2_dim ==0:
            return self.f1(x)
        elif self.f1_dim==0 and  self.f2_dim >0:
            return self.f2(x)
        else:
            x= torch.cat([self.f1(x),self.f2(x)], 1)
            return x

    def forward(self, X,taus):
        fX = self.f_automatic(X)
        return self.auto_net(fX,taus)


class AutoNet(nn.Module):
    def __init__(self,  device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234,*args, **kwargs):
        super().__init__()
        self.theta_dim =theta_dim
        self.nets = nn.ModuleList([ MLP(device=device,
                                        dim=1+i+x_dim, z_dim=1,
                                        leaky=leaky, factor=factor,
                                        n_layers=n_layers)
                                        for i in range(theta_dim)])
        self.np_random = np.random.RandomState(seed)
        self.device=device
        self.to(device)
    def forward(self, X, taus=None):
        if taus is None: # taus: batch_ size x theta_dim
            taus = self.np_random.rand(X.shape[0], self.theta_dim)
            taus = torch.from_numpy(taus).float().to(self.device)

        theta_samples=[]
        for i, net in enumerate(self.nets):
            input = torch.cat([taus[:,i:i+1]]+theta_samples + [X], dim=1) 
            theta_samples.append(net(input))

        return torch.cat(theta_samples, dim=1)

class AutoReg():
    def __init__(self, simulator, theta_dim,  x_dim, x_length,parallel=False,
                 f_manual=False, ss_f = False, f1_dim=2, f2_dim=5, 
                 
                 leaky=0.1, factor=64, n_layers=2,
                 
                 epoch=150, batch_size = 200, n_iter=100, device="cuda",
                  seed=1234, lr=0.01, vis_every = 20,observed_data = None, true_post = None,
                 true_params=None, Xs=None,posterior_sampler =None,
                 *args, **kwargs):

        self.ss_f = ss_f
        if f_manual is not None: 
            print("manual feature map")
            self.net = AutoNet_manual(f_manual,f1_dim=f1_dim, f2_dim=f2_dim, 
                            device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed )
            
        elif ss_f:
            print("deepset feature map")
            self.net = AutoNet_ss(
                            f1_dim=f1_dim, f2_dim=f2_dim, 
                            device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        else:
            print("mlp feature map")
            self.net = AutoNet(device=device, x_dim=x_dim*x_length, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        self.lr = lr
        self.simulator = simulator
        self.epoch=epoch
        self.batch_size=batch_size
        self.n_iter=n_iter
        self.np_random = np.random.RandomState(seed)
        self.device=device
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.x_length = x_length

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
        
        if parallel and (torch.cuda.device_count() > 1):
            self.net = torch.nn.DataParallel(self.net)
            print("data parallel")
        self.net.to(device)
        
    def train(self):

        self.log = []
        for epoch in range(1, self.epoch +1):
            self.current_epoch = epoch
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr*(0.99**epoch))
            running_loss = 0.0
            for idx in range(self.n_iter):
                optimizer.zero_grad()

                Thetas, X = self.simulator(self.batch_size,
                                             np_random = self.np_random)
                Thetas = Thetas.float().to(self.device)
                
                X = X.float().to(self.device)
                if not self.ss_f:
                    X = X.view(-1,self.x_dim*self.x_length)
                taus = self.np_random.rand(self.batch_size, self.theta_dim)
                taus = torch.from_numpy(taus).float().to(self.device)

                fake_thetas= self.net(X, taus)
                diff = Thetas-fake_thetas
                loss = torch.maximum(diff*taus, diff*(taus-1)).mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            loss_cum = running_loss/self.n_iter
            sample = self.sampler(self.observed_data,300,shaper=lambda x: x)
            mmd = compute_mmd(sample, self.true_post) if self.true_post is not None else 0
            Emmd = self.mmd_val() if self.mmd_measuring else 0
            self.log.append({"loss":loss_cum, "mmd": mmd, "Emmd":Emmd})

            if epoch % self.vis_every ==0:
                print(f"Epoch {epoch}")
                try:
                    self.vis(sample)
                except:
                    print("some vis err")

    def sampler(self, X, sample_size=100, shaper=None):
        if shaper is None:
            X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1).to(self.device)
        else:
            X = shaper(X)
        taus = self.np_random.rand(sample_size, self.theta_dim)
        taus = torch.from_numpy(taus).float().to(self.device)
        train_mode = self.net.train
        self.net.eval() #eval: THE most important thing

        with torch.no_grad():
            #set_trace()
            sample = self.net(X.to(self.device), taus)
            
        if train_mode: self.net.train()
        return sample.detach().cpu()

    def mmd_val(self):
        mmds = []
        for i in range(self.Xs.shape[0]):
            true_param, observed_data = self.true_params[i],self.Xs[i]
            sim_post_sample = self.sampler(observed_data,300)
            theta, sigma_sq = self.posterior_sampler(X = observed_data)
            true_post_sample = np.stack([theta,sigma_sq],1)
            mmd_value = compute_mmd(sim_post_sample,true_post_sample)
            mmds.append(mmd_value)
        return np.mean(mmds)

    def vis(self,sample):

        n_col = 3 
        fig,axis = plt.subplots(1,n_col, figsize=(4*n_col,4))#, sharex=True, sharey=True)
        df = pd.DataFrame(self.log)
        ax = axis[0]
        df["loss"].plot(ax = ax)
        ax = axis[1]
        df["Emmd"].plot(ax = ax)
        ax = axis[2]
        df["mmd"].plot(ax = ax)

        ax = axis[3]
        ax.set_title(f"Epoch {self.current_epoch}")
        sns.kdeplot(x=sample[:,0], y=sample[:,1], ax=ax, fill=False)
        sns.kdeplot(x=self.true_post[:,0], y=self.true_post[:,1], ax=ax, fill=True)
        plt.show()

    def save(self, path):
        if isinstance(self.net, torch.nn.DataParallel):
            torch.save(self.net.module.state_dict(), path)
        else:
            torch.save(self.net .state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


class AutoReg_f(AutoReg):
    def __init__(self, simulator, f1_dim=2, f2_dim=5, #f1_layers=5,
                 epoch=150, batch_size = 200, n_iter=100, device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234, lr=0.01, *args, **kwargs):

        super().__init__(simulator, epoch=epoch, batch_size = batch_size, n_iter=n_iter,
                         device=device, x_dim=x_dim, theta_dim = theta_dim,
                           leaky=leaky, factor=factor, n_layers=n_layers, seed=seed, lr=lr)

        self.net = AutoNet_ss(
                            f1_dim=f1_dim, f2_dim=f2_dim, #f1_layers =f1_layers,
                            device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)


