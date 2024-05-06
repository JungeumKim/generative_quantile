import torch
from _data.gaussian_conjugate import posterior_sampler
from _utils.breiner_util import uniform_on_unit_ball
import matplotlib.pyplot as plt
import seaborn as sns


def unif(size,generator,  eps=1E-7, device ="cuda"):
    return torch.clamp(torch.rand(size, generator=generator).to(device), min=eps, max=1-eps)


def test(nets, n_sample, gauss, HPARAM, test_val=2.2, n_test = 500, 
         save_dir=None, show=False, close=True, sphere=False,
         seed = 1234, device="cuda", fig=None, axis=None):
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    x_test = torch.tensor([test_val for _ in range(n_sample)])
    theta, sigma_sq = posterior_sampler(X = x_test,batch_size=n_test, seed=seed, h_param = HPARAM)
    
    
    if sphere:
        U = uniform_on_unit_ball(n_test, 2, np_random=seed)
        U = torch.from_numpy(U).float().to(device)
    else:
        U = unif(size=(n_test, 2), generator=generator, device = device)
    
        if gauss:
            gauss = torch.distributions.normal.Normal(
                                            torch.tensor([0.]).to(device), 
                                            torch.tensor([1.]).to(device))
            U = gauss.icdf(U)
        
    X = torch.ones((n_test,n_sample),
                          device=device)*test_val
    
    n_col = 1+len(nets)

    if fig is None: 
        fig,axis = plt.subplots(1,n_col, figsize=(4*n_col,4), sharex=True, sharey=True)

    ax=axis[0]
    ax.set_ylabel(F"x={test_val}", fontsize=20)
    sns.kdeplot(x=theta, y=sigma_sq, ax=ax, fill=False) 
    sns.kdeplot(x=theta, y=sigma_sq, ax=ax, fill=True) 
    ax.set_title("True Posterior Sample")
    
    for i, (name,net) in enumerate(nets.items()):
        
        ax=axis[i+1]
        net.eval()
        Y_hat,_ = net.grad(U, X.unsqueeze(1), onehot=False)
        Y_hat = Y_hat.detach().cpu()
    
        sns.kdeplot(x=Y_hat[:,0], y=Y_hat[:,1], ax=ax, fill=False) 
        sns.kdeplot(x=Y_hat[:,0], y=Y_hat[:,1], ax=ax, fill=True) 
        
        ax.set_title(name)
        
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir)
    if show:
        plt.show()
    if close:
        plt.close()

def credible_test(net, n_sample, HPARAM, alphas = [0.7,0.9, 0.95],test_vector=None,
                  test_val=2.2, n_test = 500, 
                     save_dir=None, show=False, close=True, sphere=False,
                     seed = 1234, device="cuda", fig=None, axis=None, show_true=True,
                     alpha=0.3, s=1,bw_adjust=1):
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    


    U = uniform_on_unit_ball(n_test, 2, np_random=seed)
    U = torch.from_numpy(U).float().to(device)

    if test_vector is None:
        X = torch.ones((n_test,n_sample),
                          device=device)*test_val
    else:
        X = test_vector.view(-1).unsqueeze(0).repeat(n_test, 1).to(device).float()

    n_col = 1+len(alphas)+show_true
    
    if fig is None: 
        fig,axis = plt.subplots(1,n_col, figsize=(4*n_col,4), sharex=True, sharey=True)

    if show_true:
        x_test = torch.tensor([test_val for _ in range(n_sample)])
    
        theta, sigma_sq = posterior_sampler(X = x_test,batch_size=n_test, seed=seed, h_param = HPARAM)
    
        ax=axis[0]
        ax.set_ylabel(F"x={test_val}", fontsize=20)
        sns.kdeplot(x=theta, y=sigma_sq, ax=ax, fill=False) 
        sns.kdeplot(x=theta, y=sigma_sq, ax=ax, fill=True) 
        #ax.scatter(x=theta, y=sigma_sq, alpha=0.3)
        ax.set_title("True Posterior Sample")

    net.eval()
    contour_paths=[]

    for i,r in enumerate(alphas):
        ax=axis[i+show_true]
        Y_hat,_ = net.grad(U*r, X.unsqueeze(1), onehot=False)
        Y_hat = Y_hat.detach().cpu()
        ax.scatter(Y_hat[:, 0], Y_hat[:, 1], alpha=alpha, s=s)
        kde=sns.kdeplot(x=Y_hat[:,0], y=Y_hat[:,1], ax=ax, fill=False, 
                        linewidths=2, levels = 2, color="red",bw_adjust=bw_adjust) 
        #sns.kdeplot(x=Y_hat[:,0], y=Y_hat[:,1], ax=ax, fill=True) 
        path = kde.collections[-2].get_paths()[0]
        contour_paths.append(path)

        ax.set_title(fr"$\alpha$={r}")

    ax=axis[i+1+show_true]
    for path in contour_paths:
        patch = plt.Polygon(path.vertices, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(patch)
        
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir)
    if show:
        plt.show()
    if close:
        plt.close()

