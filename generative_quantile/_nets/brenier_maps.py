import torch
import torch.nn as nn
from _nets.icnn import ICNN_LastInp_Quadratic
from _nets.basic_nets import  MLP

from IPython.core.debugger import set_trace

def dual_JK(U, Y_hat, Y, X):
    alpha, beta = Y_hat # alpha(U) + beta(U)^{T}X
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
        #set_trace()
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
        #if len(x.shape)==2:
        #    x = x.unsqueeze(-1)
        shape = x.shape
        assert len(shape)==3
        #set_trace()
        phi = self.common_feature_net(x.view(-1,shape[-1])).view(x.shape[0],x.shape[1],-1).sum(1)
        out = self.next_net(phi)
        if self.bn_last:
            return self.norm(out)
        return out

class ConditionalConvexQuantile(nn.Module):
    def __init__(self, xdim, udim,
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

        if  self.f1dim>0:
            self.f1 = DeepSets(dim_x=xdim,
                              dim_ss=self.f1dim,
                              factor=factor, num_layers=f1_layers, device=device)

        if  self.f2dim>0:
             self.f2 = BiRNN(input_size=xdim,
                       hidden_size=512,
                       num_layers=1,
                       xdim=self.f2dim)

        self.device =device
        self.to(device)

    def f(self, x):
        if self.f1dim>0 and  self.f2dim ==0:
            return self.f1(x)
        elif self.f1dim==0 and  self.f2dim >0:
            return self.f2(x)
        else:
            x= torch.cat([self.f1(x),self.f2(x)], 1)
            return x

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
