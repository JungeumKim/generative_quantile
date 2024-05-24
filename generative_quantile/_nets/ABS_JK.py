"""
This is code for  Adversarial Bayesian Simulation (Yuexi Wang, Veronika Rockova), implemented by Jungeum Kim.

TODO) the data should be normalized if want to mimic the original code by Wuexi Wang.
TODO) clean up device
"""

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

        x = torch.cat([noise, context], -1)

        for layer in self.layers:
                x = self.dropout(self.activation(layer(x)))
        return x


class Critic(nn.Module):

    def __init__(self,activation = "relu",dropout = 0,input_dim=2, d_hidden = [128,128,128]):

        super().__init__()
        d_in = [input_dim] + d_hidden
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


def train(generator, critic, h_params, n_sample, batch_size, simulator, Epochs=1000,
          critic_gp_factor = 5,critic_lr = 1e-4,critic_steps = 15,generator_lr = 1e-4,
          print_every=200, device="cuda", n_iter=1000, test_iter=10):

    # setup training objects
    start_time = time()
    step = 1

    generator.to(device), critic.to(device)
    opt_generator = optim.Adam(generator.parameters(), lr=generator_lr)
    opt_critic = optim.Adam(critic.parameters(), lr=critic_lr)

    for epoch in range(Epochs):
        # train loop
        WD_train, WD_test= 0, 0
        n_critic = 0
        critic_update = True

        for iter in range(n_iter):
            x, context = simulator(h_params, n_sample,batch_size)
            x, context = x.to(device), context.to(device)
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
            x, context = simulator(h_params, n_sample,batch_size)
            x, context = x.to(device), context.to(device)
            with torch.no_grad():
                x_hat = generator(context)
                critic_x_hat = critic(x_hat, context).mean()
                critic_x = critic(x, context).mean()
                WD_test += (critic_x - critic_x_hat).item()
        WD_test /= test_iter
        # diagnostics
        if epoch % print_every == 0:
            description = "epoch {} | step {} | WD_test {} | WD_train {} | sec passed {} |".format(
            epoch, step, round(WD_test, 2), round(WD_train, 2), round(time() - start_time))
            print(description)
            t = time()

