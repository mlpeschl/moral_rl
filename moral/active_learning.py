import numpy as np
import math
import scipy.stats as st


class PreferenceLearner:
    def __init__(self, n_iter, warmup, d):
        self.n_iter = n_iter
        self.warmup = warmup
        self.d = d
        self.accept_rates = None
        self.deltas = []
        self.prefs = []

    def log_preference(self, delta, pref):
        self.deltas.append(delta)
        self.prefs.append(pref)

    def w_prior(self, w):
        if np.linalg.norm(w) <=1 and np.all(np.array(w) >= 0):
            return (2**self.d)/(math.pi**(self.d/2)/math.gamma(self.d/2 + 1))
        else:
            return 0

    def sample_w_prior(self, n):
        sample = np.random.rand(n, self.d)
        w_out = []
        for w in sample:
            w_out.append(list(w/np.linalg.norm(w)))
        return np.array(w_out)

    @staticmethod
    def f_loglik(w, delta, pref):
        return np.log(np.minimum(1, np.exp(pref*np.dot(w, delta)) + 1e-5))

    @staticmethod
    def vanilla_loglik(w, delta, pref):
        return np.log(1/(1+np.exp(-pref*np.dot(w, delta))))

    @staticmethod
    def propose_w_prob(w1, w2):
        q = st.multivariate_normal(mean=w1, cov=1).pdf(w2)
        return q

    @staticmethod
    def propose_w(w_curr):
        w_new = st.multivariate_normal(mean=w_curr, cov=1).rvs()
        return w_new

    def posterior_log_prob(self, deltas, prefs, w):
        f_logliks = []
        for i in range(len(prefs)):
            f_logliks.append(self.f_loglik(w, deltas[i], prefs[i]))
        loglik = np.sum(f_logliks)
        log_prior = np.log(self.w_prior(w) + 1e-5)

        return loglik + log_prior

    def mcmc_vanilla(self, w_init='mode'):
        if w_init == 'mode':
            w_init = [0 for i in range(self.d)]

        w_arr = []
        w_curr = w_init
        accept_rates = []
        accept_cum = 0

        for i in range(1, self.warmup + self.n_iter + 1):
            w_new = self.propose_w(w_curr)

            prob_curr = self.posterior_log_prob(self.deltas, self.prefs, w_curr)
            prob_new = self.posterior_log_prob(self.deltas, self.prefs, w_new)

            if prob_new > prob_curr:
                acceptance_ratio = 1
            else:
                qr = self.propose_w_prob(w_curr, w_new) / self.propose_w_prob(w_new, w_curr)
                acceptance_ratio = np.exp(prob_new - prob_curr) * qr
            acceptance_prob = min(1, acceptance_ratio)

            if acceptance_prob > st.uniform(0, 1).rvs():
                w_curr = w_new
                accept_cum = accept_cum + 1
                w_arr.append(w_new)
            else:
                w_arr.append(w_curr)

            accept_rates.append(accept_cum / i)

        self.accept_rates = np.array(accept_rates)[self.warmup:]

        return np.array(w_arr)[self.warmup:]


class VolumeBuffer:
    def __init__(self, auto_pref=True):
        self.auto_pref = auto_pref
        self.best_volume = -np.inf
        self.best_delta = None
        self.best_observed_returns = None
        self.best_returns = None
        self.observed_logs = []
        self.objective_logs = []

    def log_statistics(self, statistics):
        self.objective_logs.append(statistics)

    def log_rewards(self, rewards):
        self.observed_logs.append(rewards)

    @staticmethod
    def volume_removal(w_posterior, delta):
        expected_volume_a = 0
        expected_volume_b = 0
        for w in w_posterior:
            expected_volume_a += (1 - PreferenceLearner.f_loglik(w, delta, 1))
            expected_volume_b += (1 - PreferenceLearner.f_loglik(w, delta, -1))

        return min(expected_volume_a / len(w_posterior), expected_volume_b / len(w_posterior))

    def sample_return_pair(self):
        observed_logs_returns = np.array(self.observed_logs).sum(axis=0)
        rand_idx = np.random.choice(np.arange(len(observed_logs_returns)), 2, replace=False)

        # v2-Environment comparison
        #new_returns_a = observed_logs_returns[rand_idx[0]]
        #new_returns_b = observed_logs_returns[rand_idx[1]]

        # v3-Environment comparison (vase agnostic)
        new_returns_a = observed_logs_returns[rand_idx[0], 0:3]
        new_returns_b = observed_logs_returns[rand_idx[1], 0:3]

        # Reset observed logs
        self.observed_logs = []

        # Also return ground truth logs for automatic preferences
        if self.auto_pref:
            objective_logs_returns = np.array(self.objective_logs).sum(axis=0)

            # v2-Environment comparison
            #logs_a = objective_logs_returns[rand_idx[0]]
            #logs_b = objective_logs_returns[rand_idx[1]]

            # v3-Environment comparison (vase agnostic)
            logs_a = objective_logs_returns[rand_idx[0], 0:3]
            logs_b = objective_logs_returns[rand_idx[1], 0:3]

            self.objective_logs = []
            return np.array(new_returns_a), np.array(new_returns_b), logs_a, logs_b
        else:
            return np.array(new_returns_a), np.array(new_returns_b)

    def compare_delta(self, w_posterior, new_returns_a, new_returns_b, logs_a=None, logs_b=None, random=False):
        delta = new_returns_a - new_returns_b
        volume_delta = self.volume_removal(w_posterior, delta)
        if volume_delta > self.best_volume or random:
            self.best_volume = volume_delta
            self.best_delta = delta
            self.best_observed_returns = (new_returns_a, new_returns_b)
            self.best_returns = (logs_a, logs_b)

    def reset(self):
        self.best_volume = -np.inf
        self.best_delta = None
        self.best_returns = None
        self.best_observed_returns = (None, None)

