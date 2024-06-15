import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from _nets.icnn import ICNN_LastInp_Quadratic
from _nets.basic_nets import  MLP,MLP_batchnorm
from _utils.breiner_util import uniform_on_unit_ball
import matplotlib.pyplot as plt
import seaborn as sns



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
    def __init__(self, xdim, udim, x_length, f_manual = None, 
                 f1dim=0, f2dim=0, factor=16, f1_layers=2,ss_f = True,
                 a_hid=512, a_layers=3, b_hid=512,b_layers=1, lstm_hidden_size=512, lstm_num_layers=1,
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
        
        if f_manual is not None:
            print("manual feature map")
            self.batch = nn.BatchNorm1d(f1dim+f2dim)

            def f (x): 
                return self.batch(f_manual(x))
            self.f = f
            
        elif ss_f: 
            print("deepset and lstm feature map")
            if  self.f1dim>0:
                self.f1 = DeepSets(dim_x=xdim,
                              dim_ss=self.f1dim,
                              factor=factor, num_layers=f1_layers, device=device)

            if  self.f2dim>0:
                self.f2 = BiRNN(input_size=xdim,
                       hidden_size=lstm_hidden_size,
                       num_layers=lstm_num_layers,
                       xdim=self.f2dim)
                
            self.f = self.f_automatic
            
        else:
            print("mlp feature map")
            self.mlp =  MLP_batchnorm(device=device,
                         dim = xdim * x_length, z_dim = f1dim+f2dim, 
                         dropout=0.5, 
                         factor=16, n_layers=2,positive=True)
            def f (x): 
                #set_trace()
                return self.mlp(x.view(-1, xdim * x_length))
            self.f = f

            
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

    def __init__(self, simulator,theta_dim,  x_dim, x_length,thresh = -10**5, device="cuda",
                 f1_dim=1,f2_dim=1, f_manual=None, ss_f = True,
                 lstm_hidden_size=512, lstm_num_layers=1,
                 epoch=1000, batch_size = 200,
                 seed = 1234, parallel=False, lr =0.01,
                 n_iter=1000, vis_every = 20,observed_data = None, true_post = None,
                 *args, **kwargs):

        self.np_random = np.random.RandomState(seed)
        self.net = ConditionalConvexQuantile(xdim=x_dim,x_length=x_length,
                                    udim=theta_dim,
                                    ss_f = ss_f,
                                    f1dim=f1_dim,
                                    f2dim=f2_dim,
                                    f_manual = f_manual,
                                    lstm_hidden_size=lstm_hidden_size,
                                    lstm_num_layers =lstm_num_layers,
                                    a_hid=512,
                                    a_layers=3,
                                    b_hid=512,
                                    b_layers=3, device="cpu")

        if parallel and (torch.cuda.device_count() > 1):
            self.net = torch.nn.DataParallel(self.net)
            print("data parallel")

        self.net.to(device)
        self.simulator = simulator
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.theta_dim = theta_dim
        self.lr = lr
        self.n_iter = n_iter
        self.thresh = thresh
        self.current_epoch = 0
        self.vis_every= vis_every
        self.observed_data =observed_data
        self.true_post = true_post
        if true_post is None:
            self.vis_every = 999999

    def train(self):
        for epoch in range(1, self.epoch +1):
            self.current_epoch = epoch
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

                for p in list(self.net.parameters()):
                    if hasattr(p, 'be_positive'):
                        p.data = p.data.clip(min=self.thresh)
        
            print('%.5f' %(running_loss))

            if epoch % self.vis_every ==0:
                try:
                    self.vis()
                except:
                    print("some vis err")

            
    def sampler(self, X, sample_size=100,r=1,shaper = None):

        u = uniform_on_unit_ball(sample_size, self.theta_dim,
                                 np_random = self.np_random)
        u = torch.from_numpy(u).float().to(self.device)
        if shaper is None:
            X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1).to(self.device).unsqueeze(-1)
        else:
            X = shaper(X)
        train_mode = self.net.train
        self.net.eval() #eval: THE most important thing
        sample = self.net.grad(u*r, X)
        if train_mode: self.net.train()
        return sample.detach().cpu()

    def vis(self,n_test=300):

        sample = self.sampler(self.observed_data,n_test)

        n_col = 1
        fig,axis = plt.subplots(1,n_col, figsize=(4*n_col,4))#, sharex=True, sharey=True)
        ax = axis
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
