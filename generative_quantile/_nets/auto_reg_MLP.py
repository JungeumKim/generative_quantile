from _nets.basic_nets import  MLP
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

class AutoNet(nn.Module):
    def __init__(self,  device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234):
        super().__init__()
        self.theta_dim =theta_dim
        self.nets = nn.ModuleList([ MLP(device=device,
                                        dim=1+i+x_dim, z_dim=1,
                                        leaky=leaky, factor=factor,
                                        n_layers=n_layers)
                                        for i in range(theta_dim)])
        self.np_random = np.random.RandomState(seed)

    def forward(self, X, taus=None):
        if taus is None: # taus: batch_ size x theta_dim
            taus = self.np_random.rand(self.batch_size, self.theta_dim)
            taus = torch.from_numpy(taus).float().to(self.device)

        theta_samples=[]
        for i, net in enumerate(self.nets):
            input = torch.cat([taus[:,i]]+theta_samples + [X], dim=1) #TODO) check whatappens, thetas =[] or has only one tensor.
            theta_samples.append(net(input))

        return torch.cat(theta_samples, dim=1)


class AutoReg(nn.Module):
    def __init__(self, simulator, epochs=1000, batch_size = 200, n_iter=100, device="cuda", x_dim=2, theta_dim = 2,
                 leaky=0.1, factor=64, n_layers=2, seed=1234):
        self.net = AutoNet(device=device, x_dim=x_dim, theta_dim = theta_dim,
                            leaky=leaky, factor=factor, n_layers=n_layers, seed=seed)
        self.simulator = simulator
        self.epochs=epochs
        self.batch_size=batch_size
        self.n_iter=n_iter

    def train(self):
        for epoch in range(1, self.epochs +1):
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

    def sampler(self, X, sample_size=100):
        X = X.view(1, -1).repeat(sample_size, 1)
        sample = self.net(X)
        return sample.detach().cpu()
