from IPython.core.debugger import set_trace
from _nets.basic_nets import  MLP
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from _nets.brenier_maps import DeepSets,BiRNN


class AutoNet_ss(nn.Module):
    def __init__(self,  f1dim=2, f2dim=2, device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234,f1_layers =3,*args, **kwargs):
        super().__init__()
        self.f1dim = f1dim
        self.f2dim= f2dim
        if  self.f1dim>0:
            self.f1 = DeepSets(dim_x=x_dim,
                              dim_ss=self.f1dim,
                              factor=factor, num_layers=f1_layers, device=device)
        if  self.f2dim>0:
            self.f2 = BiRNN(input_size=x_dim,
                   hidden_size=512,
                   num_layers=1,
                   xdim=self.f2dim)

        self.f = self.f_automatic
        self.auto_net = AutoNet(device="cuda", x_dim=f1dim+f2dim, theta_dim = theta_dim,
                                leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        self.device=device
        self.to(device)

    def f_automatic(self, x):
        if self.f1dim>0 and  self.f2dim ==0:
            return self.f1(x)
        elif self.f1dim==0 and  self.f2dim >0:
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
    def __init__(self, simulator, epoch=150, batch_size = 200, n_iter=100, device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234, lr=0.01, ss_f = False, f1dim=2, f2dim=5, f1_layers=5,
                 *args, **kwargs):
        if ss_f:
            self.net = AutoNet_ss(
                            f1dim=f1dim, f2dim=f2dim, f1_layers =f1_layers,
                            device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        else:
            self.net = AutoNet(device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        self.lr = lr
        self.simulator = simulator
        self.epoch=epoch
        self.batch_size=batch_size
        self.n_iter=n_iter
        self.np_random = np.random.RandomState(seed)
        self.device=device
        self.theta_dim = theta_dim
        
    def train(self):
        for epoch in range(1, self.epoch +1):
            print(f"Epoch {epoch}")
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr*(0.99**epoch))
            running_loss = 0.0
            for idx in range(self.n_iter):
                optimizer.zero_grad()

                Thetas, X = self.simulator(self.batch_size,
                                             np_random = self.np_random)
                Thetas = Thetas.float().to(self.device)
                X = X.float().to(self.device)
                taus = self.np_random.rand(self.batch_size, self.theta_dim)
                taus = torch.from_numpy(taus).float().to(self.device)

                fake_thetas= self.net(X, taus)
                diff = Thetas-fake_thetas
                loss = torch.maximum(diff*taus, diff*(taus-1)).mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('%.5f' %(running_loss))

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
            sample = self.net(X.to(self.device), taus)
            
        if train_mode: self.net.train()
        return sample.detach().cpu()


    def save(self, path):
        # Save the state dictionaries of generator and critic
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


class AutoReg_f(AutoReg):
    def __init__(self, simulator, f1dim=2, f2dim=5, f1_layers=5,
                 epoch=150, batch_size = 200, n_iter=100, device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234, lr=0.01, *args, **kwargs):

        super().__init__(simulator, epoch=epoch, batch_size = batch_size, n_iter=n_iter,
                         device=device, x_dim=x_dim, theta_dim = theta_dim,
                           leaky=leaky, factor=factor, n_layers=n_layers, seed=seed, lr=lr)

        self.net = AutoNet_ss(
                            f1dim=f1dim, f2dim=f2dim, f1_layers =f1_layers,
                            device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)

