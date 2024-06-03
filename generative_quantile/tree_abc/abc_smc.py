from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class ABCR_wrapper:
    def __init__(self, tree_simulator,support,observed_data,budget = 10000,*args, **kwargs):

        self.abcr= ABCR(budget, support, tree_simulator.prior_sim,
                        tree_simulator.discrepancy,
                        observed_data)

    def train(self):
        self.abcr.inference()

    def sampler(self,batch_size=100, n_pool = 10000 ):

        epsilon = np.quantile(self.abcr.discrepancies[:n_pool], 0.1)

        samples = self.abcr.posterior(epsilon)

        return samples[:batch_size]

    def save(self, path):
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self,path):
        import pickle
        with open(path, 'rb') as file:
            saved = pickle.load(file)
        self.abcr = saved.abcr


class ABCR:
    """Rejection sampling ABC object"""

    def __init__(self, budget, support, prior_sim, discrepancy, xobs, *args, **kwargs):
        """Fits rejection sampling ABC on problem instance"""
        self.support = support
        self.budget = budget
        self.prior_sim = prior_sim
        self.discrepancy = discrepancy
        self.xobs = xobs

    def inference(self):
        self.thetas = self.prior_sim(self.budget)
        self.discrepancies = self.discrepancy(self.thetas, self.xobs)

    def posterior(self, epsilon, n_evaluations=None):
        if n_evaluations is None:
            neval_ = self.budget
        else:
            neval_ = n_evaluations

        thetas_ = self.thetas[:neval_, :]
        discreps_ = self.discrepancies[:neval_]

        respost = thetas_[discreps_ < epsilon]

        if thetas_.shape[1] == 1:
            respost = respost.reshape(-1, 1)

        return respost


class ABCSMC_wrapper:
    def __init__(self, tree_simulator,support,observed_data,budget = 10000,
                 eps_discount_factor = 0.9,n_repeats = 100, initial_eps = 0.2,*args, **kwargs):


        rabc = ABCR(budget, support, tree_simulator.prior_sim,
                        tree_simulator.discrepancy, observed_data)
        rabc.inference()
        eps_init = np.quantile(rabc.discrepancies, initial_eps)

        eps_sched = eps_init * (eps_discount_factor ** np.arange(n_repeats))

        self.smc= ABCSMC(budget, support, tree_simulator.prior_sim,
                        tree_simulator.discrepancy,
                        observed_data,
                         tree_simulator.prior_density,
                         eps_sched)

    def train(self):
        self.smc.inference()

    def sampler(self,batch_size=100, n_pool = 10000 ):

        return self.smc.posterior()[:batch_size]

    def save(self, path):
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(self,path):
        import pickle
        with open(path, 'rb') as file:
            saved = pickle.load(file)
        self.smc = saved.smc

class ABCSMC():
    """Sequential Monte Carlo/Population Monte Carlo ABC object"""

    def __init__(
        self,
        budget,
        support,
        prior_simulator,
        discrepancy,
        xobs,
        prior_density,
        epsilon_sch,
        desired_post_size=1000,
        verbose=False,
        *args,
        **kwargs,
    ):
        # self.initial_batch_size = initial_batch_size
        self.n_batches = epsilon_sch.shape[0]

        self.dim = support.shape[0]
        self.support = support
        self.prior_simulator = prior_simulator
        self.prior_density = prior_density
        self.discrepancy = discrepancy
        self.desired_post_size = desired_post_size
        self.n_eval_history = np.zeros(self.n_batches)
        self.budget = budget

        self.epsilon_sch = epsilon_sch

        self.xobs = xobs

        self.verbose = verbose

        #self.inference()

    def inference(self):
        """Performs likelihood-free inference on the given problem instance

        The problem instance is characterized by the prior (support, prior_simulator, prior_density),
        the simulator, discrepancy metric, epsilon, and observed data xobs
        """

        # Initial batch, handled separately
        self.n_evaluations = 0
        n_acc = 0

        self.posteriors = []
        self.maps = []
        self.tvalues = []

        self.post_history = np.zeros((self.n_batches, self.desired_post_size, self.dim))

        self.thetas = np.empty((self.budget, self.dim))
        self.discrepancies = np.empty(self.budget)
        # self.extra = np.empty(self.desired_post_size)

        # self.extras = []

        self.thetasb = np.empty((self.desired_post_size, self.dim))
        self.w = np.ones(self.desired_post_size) / self.desired_post_size

        while n_acc < self.desired_post_size and self.n_evaluations < self.budget:
            theta = self.prior_simulator(1)
            self.thetas[self.n_evaluations] = theta

            discrep = self.discrepancy(theta, self.xobs)
            # discrep, _ = self.discrepancy(theta, self.xobs)
            self.discrepancies[self.n_evaluations] = discrep

            self.n_evaluations += 1

            if discrep < self.epsilon_sch[0]:
                self.thetasb[n_acc] = theta
                n_acc += 1

        if self.verbose:
            print(f"Batch 1 done, {self.n_evaluations} sims done")

        is_inds = np.random.choice(self.desired_post_size, self.desired_post_size, replace=True, p=self.w)
        self.posteriors.append(self.thetasb[is_inds])
        self.tvalues.append(self.n_evaluations)

        self.post_history[0] = self.thetasb
        self.n_eval_history[0] = self.n_evaluations

        # Remaining batches

        new_thetas = np.empty((self.desired_post_size, self.dim))
        new_w = np.zeros(self.desired_post_size)

        self.c_eps = 0
        for b in range(1, self.n_batches):
            prevcov = 2 * np.cov(self.thetasb.T)

            n_acc = 0
            while n_acc < self.desired_post_size and self.n_evaluations < self.budget:
                while True:
                    new_theta = self.thetasb[
                        np.random.choice(self.thetasb.shape[0], 1, p=self.w), :
                    ] + stats.multivariate_normal.rvs(mean=np.zeros(self.dim), cov=prevcov, size=1)

                    in_lower_bounds = self.support[:, 0] < new_theta
                    in_upper_bounds = self.support[:, 1] > new_theta

                    if np.all(in_lower_bounds & in_upper_bounds):
                        break
                    # if np.array([self.support[a,0] < new_theta[0,a] and self.support[a,1] > new_theta[0,a] for a in np.arange(self.dim)]).all():
                    #     break

                self.thetas[self.n_evaluations] = new_theta
                discrep = self.discrepancy(new_theta, self.xobs)
                # discrep, latest_extra = self.discrepancy(new_theta, self.xobs)

                self.discrepancies[self.n_evaluations] = discrep

                self.n_evaluations += 1

                if self.n_evaluations >= self.budget:
                    return

                if discrep < self.epsilon_sch[b]:
                    new_thetas[n_acc] = new_theta
                    # self.extra[n_acc] = latest_extra
                    denom_w = 0

                    # denom_w = self.w @ np.apply_along_axis(lambda mn: stats.multivariate_normal.pdf(new_theta, mean=mn, cov=2*np.cov(self.thetas.T)),
                    #                             axis=1,
                    #                             arr=self.thetas)

                    d = self.dim
                    inv_cov = np.linalg.inv(prevcov)
                    det_cov = np.linalg.det(prevcov)

                    # Compute difference between data point and means
                    diff = new_theta - self.thetasb

                    # Compute the exponent part of the formula
                    exponent = np.einsum("ij,ij->i", diff @ inv_cov, diff)

                    # Combine everything to get the multivariate normal PDF values
                    pdf_values = (1 / (2 * np.pi) ** (d / 2) / np.sqrt(det_cov)) * np.exp(-0.5 * exponent)

                    denom_w = np.sum(self.w * pdf_values)
                    # for i in range(self.desired_post_size):
                    #     denom_w += self.w[i] * stats.multivariate_normal.pdf(new_theta, mean=self.thetasb[i], cov=2*np.cov(self.thetasb.T))
                    new_w[n_acc] = self.prior_density(new_theta) / denom_w
                    n_acc += 1

            self.thetasb = new_thetas
            self.w = new_w / np.sum(new_w)

            is_inds = np.random.choice(self.desired_post_size, self.desired_post_size, replace=True, p=self.w)
            self.post_history[b] = self.thetasb[is_inds]
            self.n_eval_history[b] = self.n_evaluations

            self.post = self.thetasb[is_inds]
            self.posteriors.append(self.post)
            # self.extras.append(self.extra[is_inds])
            self.tvalues.append(self.n_evaluations)
            self.c_eps = b
            if self.verbose:
                print(f"Batch {b+1} done, {self.n_evaluations} sims done")

    def posterior(self, eps=None, n=None):
        """
        Returns an empirical measure approximating the posterior after n draws
        n must be less than or equal to n_evaluations in the previous call to inference
        """

        if n is None:
            n = self.n_evaluations

        if eps is None:
            eps = self.epsilon_sch[self.c_eps]

        thetas_ = self.thetasb

        imp_ = self.w[:n]

        n_s = thetas_.shape[0]

        p_n = imp_ / np.sum(imp_)
        is_indices = np.random.choice(n_s, n_s, replace=True, p=p_n)

        return thetas_[is_indices]
