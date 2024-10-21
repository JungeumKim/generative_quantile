"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), implemented by Jungeum Kim.

TODO) the data should be normalized if want to mimic the original code by Wuexi Wang.
TODO) clean up device

TODO) JK2: use all my network designs and learning schedule for training. Adam update principle. and, my learning rate.
"""

from IPython.core.debugger import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time


class Generator(nn.Module):

    def __init__(self, d_hidden = [128,128,128], x_dim = 2, z_dim =3, cond_dim=2,dropout = 0.1,
                 activation = "relu"):
        super().__init__()
        self.d_noise = z_dim
        self.x_dim =x_dim
        self.d_cond = cond_dim
        d_in = [self.d_noise + self.d_cond] +d_hidden
        d_out = d_hidden + [x_dim ]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.leaky_relu


    def forward(self, context, noise = None):
        # context: conditioning variable.
        if noise is None:
            noise = torch.randn(context.size(0), self.d_noise).to(context.device)
        #set_trace()
        x = torch.cat([noise, context], -1)

        for layer in self.layers:
                x = self.dropout(self.activation(layer(x)))
        return x


class Critic(nn.Module):

    def __init__(self,activation = "relu",dropout = 0,input_dim=2, d_cond = 4,d_hidden = [128,128,128]):

        super().__init__()
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


class BGAN():

    def __init__(self, simulator, theta_dim, x_dim, x_length,
                 device="cuda",epoch=300, batch_size = 200, d_hidden=128,
                 critic_lr=0.001, generator_lr = 0.001,
                 seed=1234, *args, **kwargs):


        self.generator = Generator(d_hidden = [d_hidden,d_hidden,d_hidden],
                                   x_dim = theta_dim,
                                   z_dim = theta_dim,
                                   cond_dim=x_dim*x_length,
                                   dropout = 0.1,
                                   activation = "relu")

        self.critic = Critic(activation = "relu",
                             dropout = 0,
                             input_dim=theta_dim,
                             d_cond = x_dim*x_length,
                             d_hidden = [d_hidden,d_hidden,d_hidden])

        self.np_random = np.random.RandomState(seed)
        self.generator.to(device), self.critic.to(device)
        self.simulator = simulator
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.x_dim=x_dim
        self.x_length=x_length
        self.critic_lr = critic_lr
        self.generator_lr = generator_lr

    def train(self, 
              critic_gp_factor = 5,
              critic_steps = 15,
              n_iter=1000, start_epoch=1, end_epoch=None):
        
        if end_epoch==None: end_epoch = self.epoch
            
        for epoch in range(start_epoch, end_epoch+1):
            print(f"Epoch {epoch}")
            opt_generator = optim.Adam(self.generator.parameters(), lr=self.generator_lr*(0.99**epoch))
            opt_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr*(0.99**epoch))

            # train loop
            WD_train, WD_test= 0, 0
            n_critic = 0
            critic_update = True

            for iter in range(n_iter):
                x, context = self.simulator(batch_size = self.batch_size,np_random = self.np_random)
                x, context = x.to(self.device), context.to(self.device)
                if len(context.shape)==3:
                    context = context.view(-1,self.x_dim*self.x_length)
                self.generator.zero_grad()
                self.critic.zero_grad()
                #set_trace()
                x_hat = self.generator(context)
                critic_x_hat = self.critic(x_hat, context).mean()


                if n_critic < critic_steps:
                    critic_x = self.critic(x, context).mean()
                    WD = critic_x - critic_x_hat
                    loss = - WD
                    loss += critic_gp_factor * self.critic.gradient_penalty(x, x_hat, context)
                    loss.backward()
                    opt_critic.step()
                    WD_train += WD.item()
                    n_critic += 1

                else: #generator 1 step.
                    loss = - critic_x_hat
                    loss.backward()
                    opt_generator.step()
                    n_critic = 0 # now, the critic will again be trained.

            WD_train /= n_iter
            self.loss_cum = WD_train
            
            # test loop

    def sampler(self, X, sample_size, shaper = None):

        if shaper is None:
            X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1)
        else: 
            X = shaper(X)
        
        X = X.to(self.device)
        with torch.no_grad():
            return self.generator(X).to("cpu")


    def save(self, path):
        # Save the state dictionaries of generator and critic
        torch.save({
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        saved = torch.load(path)
        self.generator.load_state_dict(saved['generator'])
        self.critic.load_state_dict(saved['critic'])
