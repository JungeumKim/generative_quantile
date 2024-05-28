from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

import numpy as np
from tree_abc.tree_utils import TreeProposal, CARTTree
from tree_abc.abc_smc import ABCR
from sklearn import tree
from scipy import optimize

#from tqdm import tqdm
np.seterr(invalid='ignore')

def q_A(p,pi,A):
    q = np.sqrt((p*pi)/((2*A) - (p/pi)))
    return q/np.sum(q)
def omega(q, p, pi):
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    # return np.sum(q * p / pi) / np.sum(pi * p / q)
    return np.sum(np.where(pi != 0, q*p /pi, 0)) / np.sum(np.where(q != 0, pi*p/q, 0))

def omega_A(p, pi, A):
    return omega(q_A(p,pi,A), p, pi)

def RULE_OPTIMAL_MAX_A(prior_probs, lh_probs):
    p = prior_probs * lh_probs / np.sum(prior_probs * lh_probs)
    pi = prior_probs / np.sum(prior_probs)

    def target(A):
        return -1 * omega_A(p, pi, A)

    bnds = [0.5*np.max(p/pi), np.max(p/pi)]

    res = optimize.minimize_scalar(target, 0.75*np.max(p/pi), bounds=bnds, method="bounded")
    A_star = res.x

    return q_A(p,pi,A_star)

def RULE_POSTERIOR(prior_probs, lh_probs):
    return prior_probs * lh_probs

class TreeABC_wrapper:
    def __init__(self, tree_simulator,support,observed_data,budget = 10000,
                 eps_discount_factor = 0.9,n_repeats = 100, initial_eps = 0.2,
                 verbose=True):

        rabc = ABCR(budget, support, tree_simulator.prior_sim,
                        tree_simulator.discrepancy, observed_data)
        eps_init = np.quantile(rabc.discrepancies, initial_eps)
        eps_sched = eps_init * (eps_discount_factor ** np.arange(n_repeats))

        self.tabc = TreeABC(budget, support, tree_simulator.prior_sim,
                        tree_simulator.discrepancy,
                        observed_data,
                        tree_simulator.prior_density,
                        eps_sched,
                        verbose=verbose,
                        proposal_update_rule=RULE_OPTIMAL_MAX_A, #treeproposal.RULE_POSTERIOR,
                        n_leaves=1000)

    def train(self):
        self.tabc.inference()

    def sampler(self,batch_size=100, n_proposal=5000):
        return self.tabc.sampler(batch_size, n_proposal)

class TreeABC:
    def __init__(
        self,
        budget,
        support,
        prior_simulator,
        discrepancy,
        xobs,
        prior_density,
        epsilon_sch,
        verbose=False,
        desired_proposal_n_successes=1000,
        n_leaves=1000,
        maxdepth=6,
        n_trees_bart=10,
        proposal_update_rule= RULE_OPTIMAL_MAX_A, #RULE_POSTERIOR, #RULE_OPTIMAL_MAX_A,
        save_every=1000,
        max_points_to_fit = None,
        *args,
        **kwargs,
    ):
        self.n_leaves = n_leaves
        self.rule = proposal_update_rule
        self.desired_proposal_n_successes = desired_proposal_n_successes
        self.budget = budget
        self.n_evaluations = 0
        self.save_every = save_every
        self.dim = support.shape[0]
        self.support = support
        self.prior_simulator = prior_simulator
        self.prior_density = prior_density
        self.discrepancy = discrepancy
        self.epsilon_sch = epsilon_sch
        self.maxdepth = maxdepth
        self.n_trees_bart = n_trees_bart
        self.xobs = xobs
        self.verbose = verbose
        self.posteriors = []
        self.tvalues = []
        self.max_points_to_fit = max_points_to_fit
        print(f"MPTF: {max_points_to_fit}")

    def inference(self):
        """Performs likelihood-free inference on the given problem instance

        The problem instance is characterized by the prior (support, prior_simulator, prior_density),
        the simulator, discrepancy metric, epsilon, and observed data xobs
        """

        # Initial batch, handled separately

        bsz = self.desired_proposal_n_successes

        self.thetas = np.zeros((self.budget, self.dim))
        self.discrepancies = np.zeros(self.budget)
        self.extra = np.zeros(self.budget)
        self.extras = []
        self.acceptances = np.zeros(self.budget)
        self.q = np.zeros(self.budget)
        self.importances = np.zeros(self.budget)
        self.c_eps = 0

        self.proposal_info = {}

        #pbar = tqdm(total=self.budget)

        n_accepted = 0
        t = 0
        while n_accepted < self.desired_proposal_n_successes and t < self.budget:
            self.thetas[t] = self.prior_simulator(1)
            self.discrepancies[t] = self.discrepancy(self.thetas[t].reshape(1, -1), self.xobs)
            self.acceptances[t] = (self.discrepancies[t] < self.epsilon_sch[0]) * 2 - 1
            self.q[t] = self.prior_density(self.thetas[t].reshape(1, -1))[0]
            self.importances[t] = 1.0

            if self.acceptances[t] == 1:
                n_accepted += 1
            t += 1
            #pbar.update(1)

        if self.verbose:
            print(f"Batch 1 done, {t} sims done. Current epsilon = {self.epsilon_sch[0]}")

        theta_now = self.thetas[:t]
        discr_now = self.discrepancies[:t]
        imports_now = self.importances[:t]

        thlp = theta_now[discr_now < self.epsilon_sch[self.c_eps]][-bsz:]
        wlp = imports_now[discr_now < self.epsilon_sch[self.c_eps]][-bsz:]

        wlp = wlp / np.sum(wlp)
        plp_inds = np.random.choice(bsz, bsz, replace=True, p=wlp)
        self.posteriors.append(thlp[plp_inds])
        self.tvalues.append(t)

        self.c_eps = 1
        while t < self.budget:
            n_accepted = 0

            current_acceptances = (self.discrepancies[: (t - 1)] < self.epsilon_sch[self.c_eps - 1]) * 2 - 1
            y = (current_acceptances+1) / 2
            if self.max_points_to_fit is not None and self.max_points_to_fit < t:
                sampled_inds = np.random.choice(len(self.thetas[:(t-1)]), size=self.max_points_to_fit, replace=False)
                X_fit = self.thetas[:(t-1)][sampled_inds]
                current_acceptances_fit = current_acceptances[sampled_inds]
            else:
                X_fit = self.thetas[:(t-1)]
                current_acceptances_fit = current_acceptances

            maxdepth = self.maxdepth

            tree_model = tree.DecisionTreeClassifier(
                max_depth=maxdepth,
                max_leaf_nodes=np.min((self.n_leaves)),
                min_samples_leaf=10)

            tree_model.fit(X_fit, current_acceptances_fit)
            tree_obj = CARTTree(self.thetas[:(t-1)], y, tree_model.tree_, self.support)
            TP = TreeProposal(tree_obj, self.support, self.prior_density, self.rule)
            #
            disc = 1.0
            TP.accs /= disc
            TP.rejs /= disc

            print(len(TP.boundaries))

            self.proposal_info[t] = {"bounds": TP.boundaries, "probs": TP.probs}

            while n_accepted < self.desired_proposal_n_successes and t < self.budget:

                self.thetas[t], bin_idx = TP.sample(1, returnbins=True)
                self.discrepancies[t] = self.discrepancy(self.thetas[t].reshape(1, -1), self.xobs)
                self.acceptances[t] = (self.discrepancies[t] < self.epsilon_sch[self.c_eps]) * 2 - 1
                self.q[t] = TP.density(self.thetas[t].reshape(1, -1))[0]
                self.importances[t] = self.prior_density(self.thetas[t])[0] / self.q[t]

                if self.acceptances[t] == 1:
                    n_accepted += 1

                # update TP now
                TP.update(bin_idx, self.discrepancies[t] < self.epsilon_sch[self.c_eps], n_add=1)
                t += 1
                #pbar.update(1)
                if t % self.save_every == 0:
                    self.proposal_info[t] = {"bounds": TP.boundaries, "probs": TP.probs}

            if self.verbose:
                print(f"Batch {self.c_eps+1} done, {t} sims done. Current epsilon = {self.epsilon_sch[self.c_eps]}")

            if n_accepted < bsz:
                break

            theta_now = self.thetas[:t]
            discr_now = self.discrepancies[:t]
            imports_now = self.importances[:t]
            extra_now = self.extra[:t]
            elp = extra_now[discr_now < self.epsilon_sch[self.c_eps]][-bsz:]

            thlp = theta_now[discr_now < self.epsilon_sch[self.c_eps]][-bsz:]
            wlp = imports_now[discr_now < self.epsilon_sch[self.c_eps]][-bsz:]
            wlp = wlp / np.sum(wlp)
            plp_inds = np.random.choice(bsz, bsz, replace=True, p=wlp)
            self.posteriors.append(thlp[plp_inds])
            self.extras.append(elp[plp_inds])
            self.tvalues.append(t)

            self.c_eps += 1

        #pbar.close()
        self.tree_model = tree_model
        self.tree_obj = tree_obj
        self.TP = TP

    def posterior(self, eps=None, n=None):
        """Returns an empirical measure approximating the posterior after n draws
        n must be less than or equal to n_evaluations in the previous call to inference
        """
        if n is None:
            n = self.budget
        if eps is None:
            eps = self.epsilon_sch[self.c_eps]
        thetas_ = self.thetas[:n]
        accepts_ = (self.discrepancies[:n] < eps) * 2 - 1
        imp_ = self.importances[:n]

        n_s = thetas_[accepts_ == 1].shape[0]

        p_n = imp_[accepts_ == 1] / np.sum(imp_[accepts_ == 1])

        is_indices = np.random.choice(n_s, n_s, replace=True, p=p_n)

        return thetas_[accepts_ == 1][is_indices]


    def sampler(self, batch_size=100, n_proposal=5000):

        theta_now, _ = self.TP.sample(n_proposal, returnbins=True) #: (n_proposal, dim)
        #discr_now = self.discrepancy(theta_now, self.xobs) # from (n_proposal, dim) to (n_proposal,)
        q = self.TP.density(theta_now).reshape(-1)
        imports_now = self.prior_density(self.thetas).reshape(-1)  / q # (n_proposal, ),  (n_proposal, ) -> (n_proposal, )

        # importance resampling
        #order = discr_now.argsort()
        thlp = theta_now#[order][-batch_size:]
        wlp = imports_now#[order][-batch_size:]
        wlp = wlp / np.sum(wlp)
        plp_inds = np.random.choice(len(wlp), batch_size,  replace=True, p=wlp)
        
        #set_trace()
        #print(thlp[plp_inds])
        return thlp[plp_inds]


