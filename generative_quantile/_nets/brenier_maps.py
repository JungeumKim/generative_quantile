import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from _nets.icnn import ICNN_LastInp_Quadratic
from _nets.basic_nets import  MLP
from _utils.breiner_util import uniform_on_unit_ball


from IPython.core.debugger import set_trace

def dual_JK(U, alpha, beta, Y, X):
    # alpha(U) + beta(U)^{T}X
    # alpha: n x 1, beta: n x d, X: n x d,
    Y = Y.permute(1, 0) #d x n
    X = X.permute(1, 0) # d x n
    # beta: n x d, X: d x n,
    BX = torch.mm(beta, X) # n x n

    # U: n x d  , Y: d x n
    UY = torch.mm(U, Y) # n x n
    psi = UY - alpha - BX # n x n - n x 1 - n x n
    sup, _ = torch.max(psi, dim=0) # n

    #Corrected loss: 2024/May/2
    loss = torch.mean(alpha)
    loss += sup.mean()

    return loss

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, xdim, bn_last=True, device="cuda"):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bn_last = bn_last
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, xdim)
        self.norm = nn.BatchNorm1d(xdim, momentum=1.0, affine=False)
        self.device = device
        self.to(device)

    def forward(self, x):
        '''
        input x: (batch_size x T x dim_x)
        '''
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        if self.bn_last:
            return self.norm(out)
        return out

class DeepSets(nn.Module):
    def __init__(self, dim_x, dim_ss, factor=16, num_layers=2, device="cuda", bn_last=True):
        super(DeepSets, self).__init__()

        self.common_feature_net = MLP(device=device,
                                      dim=dim_x,
                                      z_dim = dim_ss,
                                      factor=factor,
                                      n_layers=num_layers)

        self.next_net = MLP(device=device,
                            dim=dim_ss,
                            z_dim = dim_ss,
                            factor=factor,
                            n_layers=num_layers)

        self.to(device)
        self.device = device
        self.bn_last = bn_last
        self.norm = nn.BatchNorm1d(dim_ss, momentum=1.0, affine=False)

    def forward(self, x):
        shape = x.shape
        assert len(shape)==3
        phi = self.common_feature_net(x.view(-1,shape[-1])).view(x.shape[0],x.shape[1],-1).sum(1)
        out = self.next_net(phi)
        if self.bn_last:
            return self.norm(out)
        return out

class ConditionalConvexQuantile(nn.Module):
    def __init__(self, xdim, udim, f_manual = None,
                 f1dim=0, f2dim=0, factor=16, f1_layers=2,
                 a_hid=512, a_layers=3, b_hid=512,b_layers=1,
                 device="cuda"):

        super(ConditionalConvexQuantile, self).__init__()
        self.f1dim = f1dim
        self.f2dim= f2dim

        self.alpha = ICNN_LastInp_Quadratic(input_dim=udim,
                                    hidden_dim=a_hid,
                                    activation='celu',
                                    num_layer=a_layers)
        self.beta = ICNN_LastInp_Quadratic(input_dim=udim,
                                    hidden_dim=b_hid,
                                    activation='celu',
                                    num_layer=b_layers,
                                    out_dim=self.f1dim+self.f2dim)
        if f_manual is None:
            if  self.f1dim>0:
                self.f1 = DeepSets(dim_x=xdim,
                              dim_ss=self.f1dim,
                              factor=factor, num_layers=f1_layers, device=device)

            if  self.f2dim>0:
                self.f2 = BiRNN(input_size=xdim,
                       hidden_size=512,
                       num_layers=1,
                       xdim=self.f2dim)
            self.f = self.f_automatic
        else:
            self.f = f_manual

        self.device =device
        self.to(device)

    def forward(self, z, x):
        alpha = self.alpha(z)
        beta = self.beta(z)
        f = self.f(x)
        return alpha, beta, f
    
    def grad(self, u, x):
        f = self.f(x)
        u.requires_grad = True
        phi = (self.alpha(u).view(-1) + (self.beta(u) * f).sum(1).view(-1)).sum()
        d_phi = torch.autograd.grad(phi, u, create_graph=True)[0]
        return d_phi

    def f_automatic(self, x):
        if self.f1dim>0 and  self.f2dim ==0:
            return self.f1(x)
        elif self.f1dim==0 and  self.f2dim >0:
            return self.f2(x)
        else:
            x= torch.cat([self.f1(x),self.f2(x)], 1)
            return x

class BayesQ():

    def __init__(self, simulator, device="cuda",
                 epoch=1000, batch_size = 200,
                 seed = 1234, parallel=False, lr =0.01,
                 n_iter=1000, theta_dim = 2,x_dim=2, f1_dim=1,f2_dim=1, f_manual=None, *args, **kwargs):

        self.np_random = np.random.RandomState(seed)
        self.net = ConditionalConvexQuantile(xdim=x_dim,
                                    udim=theta_dim,
                                    f1dim=f1_dim,
                                    f2dim=f2_dim,
                                    f_manual = f_manual,
                                    a_hid=512,
                                    a_layers=3,
                                    b_hid=512,
                                    b_layers=3, device="cpu")
        if parallel and (torch.cuda.device_count() > 1):
            self.net = torch.nn.DataParallel(self.net)
        self.net.to(device)

        self.simulator = simulator
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.theta_dim = theta_dim
        self.lr = lr
        self.n_iter = n_iter

    def train(self):
        for epoch in range(1, self.epoch +1):
            print(f"Epoch {epoch}")
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr*(0.99**epoch))
            running_loss = 0.0
            for idx in range(self.n_iter):

                Thetas, X = self.simulator(self.batch_size,
                                             np_random = self.np_random)
                Thetas = Thetas.float().to(self.device)
                X = X.float().to(self.device)

                #Thetas: batch_size x theta_dim, X: batch_size x n_sample
                # X later changes to batch_size x x_dim x m, where m = n_sample/x_dim
                u = uniform_on_unit_ball(self.batch_size, self.theta_dim, np_random=self.np_random)
                u = torch.from_numpy(u).float().to(self.device)

                optimizer.zero_grad()
                alpha, beta, fX= self.net(u, X)
                loss = dual_JK(u, alpha, beta, Y=Thetas, X=fX)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('%.5f' %(running_loss))
            
    def sampler(self, X, sample_size=100,r=1):

        u = uniform_on_unit_ball(sample_size, self.theta_dim,
                                 np_random = self.np_random)
        u = torch.from_numpy(u).float().to(self.device)
        X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1).to(self.device).unsqueeze(-1)
        train_mode = self.net.train
        self.net.eval() #eval: THE most important thing
        sample = self.net.grad(u*r, X)
        if train_mode: self.net.train()
        return sample.detach().cpu()

    def save(self, path):
        if isinstance(self.net, torch.nn.DataParallel):

            torch.save(self.net.module.state_dict(), path)
        else:
            torch.save(self.net .state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
