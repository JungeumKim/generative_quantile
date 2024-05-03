import torch
from torch.utils import data
import numpy as np

from gen_data import make_spiral

class Synthetic(data.Dataset):
    def __init__(self, args,  n=1000, seed =1234, device="cpu"):
        self.n = n
        np.random.seed(seed)
        self.device = device
        #self.y = np.random.normal(loc=0, scale=1, size=(self.n, 1))
        #self.y = np.random.multivariate_normal(mean=[2, 3], cov=np.array([[3,2],[2,5]]), size=(self.n))

        #torch.manual_seed(0)
        self.y, self.x = make_spiral(n_samples_per_class=self.n, n_classes=3,
            n_rotations=1.5, gap_between_spiral=1.0, noise=0.2,
                gap_between_start_point=0.1, equal_interval=True)
        '''

        self.y, self.x = make_moons(n_samples=args.n, xy_ratio=2.0, x_gap=-0.2, y_gap=0.2, noise=0.1)
        '''

    def __len__(self):
        return len(self.y)#self.y

    def __getitem__(self, i):
        if torch.is_tensor(self.y):
            return self.y[i].float().to(self.device), self.x[i].to(self.device)
        y = torch.from_numpy(self.y[i]).float().to(self.device)
        x = torch.from_numpy(np.array(self.x[i])).to(self.device)
        return y, x