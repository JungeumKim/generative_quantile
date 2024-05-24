import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def uniform_on_unit_ball(size, d, np_random=None, seed=1234,surface=False):
    if np_random is None:
        np_random = np.random.RandomState(seed)
        
    
    if surface:
        r=1
    else:
        # Generate random radii uniformly between 0 and 1
        # This is the standard case.
        r = np.random.rand(size, 1) 
    # Generate random points on the unit sphere S^{d-1}
    phi = np.random.randn(size, d)
    phi = phi / np.linalg.norm(phi, axis=1, keepdims=True)
    
    # Scale the points on the sphere by the radius to get points inside the ball
    samples = r * phi
    return samples


def plot2d(Y, name=None, labels=None, show=False, ax=None, close=False,s=5):
    #Y = Y.detach().cpu().numpy()
    #labels = labels.detach().cpu().numpy().flatten()
    #fig = plt.figure(figsize=(5, 5))
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5, 5))
    
    #ax = fig.add_subplot(1, 1, 1)
    #sns.kdeplot(Y[:, 0], Y[:, 1], cmap='Blues', shade=True, thresh=0)
    sns.scatterplot(x=Y[:,0], y=Y[:,1], hue=labels, s=s, ax=ax)
    '''
    H, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], 200, range=[[-4, 4], [-4, 4]])
    plt.imshow(H.T, cmap='BuPu')
    '''
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
    
    if name is not None:
        plt.savefig("./" + name)
    if show:
        plt.show()
    if close:
        plt.clf()
    

def histogram(Y, name):
    Y = Y.detach().cpu().numpy()
    plt.hist(Y, bins=25)
    plt.savefig("./" + name)
    plt.clf()

def plotaxis(Y, name):
    y1, y2 = Y[:,0], Y[:,1]
    histogram(y1, name=str(name)+'_x1.png')
    histogram(y2, name=str(name)+'_x2.png')


