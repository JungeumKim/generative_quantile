import torch
import torch.nn as nn
import torch.nn.functional as F
from brenier.ot_modules.icnn import *
#from supp.distribution_output import *
from brenier.supp.piecewise_linear import *
from _nets.basic_nets import  MLP

from IPython.core.debugger import set_trace
#device = "cpu"# torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def dual_JK(U, Y_hat, Y, X, eps=0,efficient=False):
    alpha, beta = Y_hat # alpha(U) + beta(U)^{T}X
    # alpha: n x 1, beta: n x d, X: n x d,

    #if efficient:
    #    loss = (alpha.view(-1,1) + (beta.unsqueeze(2) * X.permute(1, 0).unsqueeze(0)).mean(-1).sum(1).view(-1,1)) # n x 1
    #    #beta.unsqueeze(2): n x d x 1
    #    #X.permute(1, 0).unsqueeze(0): 1 x d x n
    #    # their multiplication: n x d x n : mean: nxd
    #else:
    #    loss = (alpha.view(-1,1) + (beta * X).sum(1).view(-1,1)) # n x 1



    Y = Y.permute(1, 0) #d x n
    X = X.permute(1, 0) # d x n
    # beta: n x d, X: d x n,
    BX = torch.mm(beta, X) # n x n

    # U: n x d  , Y: d x n
    UY = torch.mm(U, Y) # n x n
    # (U, Y), (U, X), beta.shape(bs, nclass), X.shape(bs, nclass)
    #print(BX.shape, UY.shape, alpha.shape)
    psi = UY - alpha - BX # n x n - n x 1 - n x n

    sup, _ = torch.max(psi, dim=0) # n

    #print(sup.shape)
    #print(UY.min(), UY.max(), sup.mean())
    #Corrected loss: 2024/May/2
    loss = torch.mean(alpha)
    loss += sup.mean()


    if eps == 0:
        return loss

    l = torch.exp((psi-sup)/eps)
    loss += eps*torch.mean(l)
    return loss



def dual(U, Y_hat, Y, X, eps=0):
    alpha, beta = Y_hat # alpha(U) + beta(U)^{T}X
    Y = Y.permute(1, 0) #d x n
    X = X.permute(1, 0) # d x n
    # beta: n x d, x: d x n,
    BX = torch.mm(beta, X) # n x n
    loss = torch.mean(alpha)
    # U: n x d  , Y: d x n
    UY = torch.mm(U, Y) # n x n
    # (U, Y), (U, X), beta.shape(bs, nclass), X.shape(bs, nclass)
    #print(BX.shape, UY.shape, alpha.shape)
    psi = UY - alpha - BX # n x n - n x 1 - n x n
    sup, _ = torch.max(psi, dim=0)
    #print(sup.shape)
    #print(UY.min(), UY.max(), sup.mean())
    loss += torch.mean(sup)

    if eps == 0:
        return loss

    l = torch.exp((psi-sup)/eps)
    loss += eps*torch.mean(l)
    return loss

def dual_unconditioned(U, Y_hat, Y, eps=0):
    loss = torch.mean(Y_hat)
    Y = Y.permute(1, 0)
    psi = torch.mm(U, Y) - Y_hat
    sup, _ = torch.max(psi, dim=0)
    loss += torch.mean(sup)

    if eps == 0:
        return loss

    l = torch.exp((psi-sup)/eps)
    loss += eps*torch.mean(l)
    return loss

def generate_x():
    x = torch.zeros(40)
    with open('./description.txt') as f:
        for line in f:
            i = attributes.index(line[:-1])
            x[i] = 1
    return x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

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
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        #set_trace()
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        if self.bn_last:
            return self.norm(out), out
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
            return self.norm(out), out
        return out


class ConditionalConvexQuantile(nn.Module):
    def __init__(self, xdim, ydim, fdim, a_hid=512, a_layers=3, b_hid=512,
                 b_layers=1, device="cuda",use_f=True,deepf = False, n_layers=2, factor=16):
        super(ConditionalConvexQuantile, self).__init__()
        self.xdim = xdim
        self.fdim = fdim
        self.a_hid=a_hid
        self.a_layers=a_layers
        self.b_hid=b_hid
        self.b_layers=b_layers
        self.use_f=use_f

        self.alpha = ICNN_LastInp_Quadratic(input_dim=ydim,
                                    hidden_dim=self.a_hid,#1024,#512
                                    activation='celu',
                                    num_layer=self.a_layers)
        self.beta = ICNN_LastInp_Quadratic(input_dim=ydim,
                                    hidden_dim=self.b_hid,
                                    activation='celu',
                                    num_layer=self.b_layers,
                                    out_dim=self.fdim)

        if deepf:
            self.f = DeepSets(dim_x=xdim,
                              dim_ss=fdim, 
                              factor=factor, num_layers=n_layers, device=device)
        else:
            self.f = BiRNN(input_size=xdim,
                       hidden_size=512,
                       num_layers=1,
                       xdim=self.fdim)
        self.device =device
        self.to(device)
        #self.f = ShallowRegressionLSTM(1, 128)
        # MLP

        #self.bn1 = nn.BatchNorm1d(self.xdim, momentum=1.0, affine=False)

        #self.f = nn.BatchNorm1d(self.xdim, affine=False)

    def forward(self, z, x=None):
        # we want onehot for categorical and non-ordinal x.
        if self.xdim == 0:
            return self.alpha(z)
        alpha = self.alpha(z)
        beta = self.beta(z) #torch.bmm(self.beta(z).unsqueeze(1), self.fc_x(x).unsqueeze(-1))
        #quad = (z.view(z.size(0), -1) ** 2).sum(1, keepdim=True) / 2
        return alpha, beta #, self.fc_x(x)
    
    def grad(self, u, x=None, onehot=True):
        if self.use_f:
            if onehot and self.xdim > 0:
                x,xv = self.to_onehot(x,self.xdim)
            elif x != None:
                x, xv = self.f(x)#self.bn1(x)
        else:
            x,xv=x.squeeze(1),x.squeeze(1)
        u.requires_grad = True 
        phi = self.alpha(u).view(-1)
        if self.xdim != 0 and x != None:
            #set_trace()
            #phi += (torch.bmm(self.beta(u).unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)).sum()
            phi += (self.beta(u) * x).sum(1).view(-1)
        phi = phi.sum()
        d_phi = torch.autograd.grad(phi, u, create_graph=True)[0]
        return d_phi, xv
    '''
    def grad_multi(self, u, x):
        if x == None:
            x = generate_x()
        x_s = x.shape[-1]
        for i in range(40):
            if x[i] == 1:
                print(attributes[i], end=',')
        x = x.expand(1, x_s)
        x = x.repeat(u.shape[0], 1).float().cuda()
        x = self.f(x)
        u.requires_grad = True
        phi = self.alpha(u).sum() + (torch.bmm(self.beta(u).unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)).sum()
        d_phi = torch.autograd.grad(phi, u, create_graph=True)[0]
        return d_phi
    '''
    
    def invert(self, y):
        raise NotImplementedError
    
    def to_onehot(self, x,n_c):
        with torch.no_grad():
            onehot = torch.zeros((x.shape[0], n_c), device=self.device)
            onehot.scatter_(dim=-1, index=x.view(x.shape[0], 1), value=1)
            #onehot -= 1/self.xdim
            #onehot = self.bn1(onehot)

        #set_trace()
        if self.use_f:
            return onehot
        else:
            return self.f(onehot.unsqueeze(1))

    def weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
