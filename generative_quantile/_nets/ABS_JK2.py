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

from _nets.brenier_maps import DeepSets,BiRNN



class Auto_ss(nn.Module):
    def __init__(self,  f1dim=2, f2dim=2, device="cuda", x_dim=2,
                 factor=64, f1_layers =3,*args, **kwargs):
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

        self.device=device
        self.to(device)

    def forward(self, x):
        if self.f1dim>0 and  self.f2dim ==0:
            return self.f1(x)
        elif self.f1dim==0 and  self.f2dim >0:
            return self.f2(x)
        else:
            x= torch.cat([self.f1(x),self.f2(x)], 1)
            return x



class Generator(nn.Module):

    def __init__(self, d_hidden = [128,128,128], x_dim = 2, z_dim =3, cond_dim=2,dropout = 0.1,
                  activation = "relu", lower_bound=None, ss_f = False, f1dim=2,f2dim=2):
        super().__init__()
        self.ss_f = ss_f
        if ss_f:
            self.ss_f = Auto_ss(f1dim=f1dim, f2dim=f2dim, x_dim=cond_dim)
            cond_dim = f1dim + f2dim

        self.d_noise = z_dim
        self.x_dim =x_dim
        self.d_cond = cond_dim
        d_in = [self.d_noise + self.d_cond] +d_hidden
        d_out = d_hidden + [x_dim ]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.leaky_relu
        self.lower_bound=lower_bound

    def forward(self, context, noise = None):
        # context: conditioning variable.
        if noise is None:
            noise = torch.randn(context.size(0), self.d_noise).to(context.device)

        x = torch.cat([noise, context], -1)

        for layer in self.layers:
                x = self.dropout(self.activation(layer(x)))

        if self.lower_bound is not None:
            # it is only for the special case where the first element is known to be lowerbounded e.g., by 0.
            x[:,0]  = x[:,0].clip(min=self.lower_bound)
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


class ABS():

    def __init__(self, simulator, x_dim, theta_dim, ss_dim,
                 device="cuda",epoch=1000, batch_size = 200, seed=1234, lower_bound=None,
                 ss_f = False, f1dim=2,f2dim=2, *args, **kwargs):


        self.generator = Generator(d_hidden = [128,128,128],
                                   x_dim = theta_dim,
                                   z_dim =ss_dim,
                                   cond_dim=x_dim,
                                   dropout = 0.1,
                                   activation = "relu",
                                   lower_bound=lower_bound, ss_f = ss_f, f1dim=f2dim,f2dim=f1dim)

        self.critic = Critic(activation = "relu",
                             dropout = 0,
                             input_dim=theta_dim,
                             d_cond = x_dim,
                             d_hidden = [128,128,128])

        self.np_random = np.random.RandomState(seed)
        self.generator.to(device), self.critic.to(device)
        self.simulator = simulator
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size

    def train(self, critic_gp_factor = 5,
              critic_lr = 0.01,
              critic_steps = 15,
              generator_lr = 0.01,
              print_every=20,
              n_iter=1000,
              test_iter=10):
        generator = self.generator
        critic = self.critic
        simulator = self.simulator
        # setup training objects
        start_time = time()
        local_start_time = time()
        step = 1



        for epoch in range(self.epoch):
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
                generator.zero_grad()
                critic.zero_grad()
                x_hat = generator(context)
                critic_x_hat = critic(x_hat, context).mean()


                if n_critic < critic_steps:
                    critic_x = critic(x, context).mean()
                    WD = critic_x - critic_x_hat
                    loss = - WD
                    loss += critic_gp_factor * critic.gradient_penalty(x, x_hat, context)
                    loss.backward()
                    opt_critic.step()
                    WD_train += WD.item()
                    n_critic += 1

                else: #generator 1 step.
                    loss = - critic_x_hat
                    loss.backward()
                    opt_generator.step()
                    n_critic = 0 # now, the critic will again be trained.

                step += 1
            WD_train /= n_iter
            # test loop

            for iter_t in range(test_iter):
                x, context = simulator(batch_size = self.batch_size)
                x, context = x.to(self.device), context.to(self.device)
                with torch.no_grad():
                    x_hat = generator(context)
                    critic_x_hat = critic(x_hat, context).mean()
                    critic_x = critic(x, context).mean()
                    WD_test += (critic_x - critic_x_hat).item()
            WD_test /= test_iter
            # diagnostics
            if epoch % print_every == 0:
                description = "epoch {} | step {} | WD_test {} | WD_train {} | sec passed {} (total {}) |".format(
                epoch, step, round(WD_test, 2), round(WD_train, 2),round(time() - local_start_time),round(time() - start_time) )
                print(description)
                local_start_time = time()

    def sampler(self, X, sample_size):
        X = torch.from_numpy(X).float().view(1, -1).repeat(sample_size, 1)
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
