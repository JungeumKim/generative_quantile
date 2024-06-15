import torch
import torch.nn as nn
from IPython.core.debugger import set_trace
from scipy.stats import truncnorm
import numpy as np

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

###################### Basic accessories setup ###############################


class ConvexLinear(nn.Linear):
    def __init__(self,  *kargs, **kwargs):
        super(ConvexLinear, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0
            p =self.weight.data
            #print("p1",p)
            self.weight.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()
            #print("p2",self.weight.data)
        self.weight.data = self.weight.data.clip(min=-10**5)
        
    def forward(self, input):
        #set_trace()
        out = nn.functional.linear(input, self.weight, self.bias)
        return out

class ICNN_LastInp_Quadratic(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layer, out_dim=1):
        super(ICNN_LastInp_Quadratic, self).__init__()
        # torch.set_default_dtype(torch.float64)
        # num_layer = the number excluding the last layer
        self.activation = get_activation(activation)
        self.num_layer = num_layer

        self.w0 = torch.nn.Parameter(torch.log(torch.exp(torch.ones(1)) - 1), requires_grad=True)
        self.w1 = torch.nn.Parameter(torch.zeros(1))

        self.fc1_normal = nn.Linear(input_dim, hidden_dim, bias=True)

        # begin to define my own normal and convex and activation
        self.normal = nn.ModuleList([nn.Linear(
            input_dim, hidden_dim, bias=True) for i in range(2, self.num_layer + 1)])

        self.convex = nn.ModuleList([ConvexLinear(
            hidden_dim, hidden_dim, bias=False) for i in range(2, self.num_layer + 1)])

        self.last_convex = ConvexLinear(hidden_dim, out_dim, bias=False)
        self.last_linear = nn.Linear(input_dim, 1, bias=True)


    def forward(self, input):
        x = self.activation(self.fc1_normal(input)).pow(2)

        for i in range(self.num_layer - 1):
            x = self.activation(self.convex[i](
                x).add(self.normal[i](input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x


def get_activation(activation, leaky_relu_slope=0.6):
    if activation == 'relu':
        return nn.ReLU(True)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(leaky_relu_slope)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softsign':
        return nn.Softsign()
    elif activation == 'Prelu':
        return nn.PReLU()
    elif activation == 'Rrelu':
        return nn.RReLU(0.5, 0.8)
    elif activation == 'hardshrink':
        return nn.Hardshrink()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softsign':
        return nn.Softsign()
    elif activation == 'tanhshrink':
        return nn.Tanhshrink()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)

