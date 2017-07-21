import numpy as np
import scipy.special as sps
import scipy.stats as stats
from sampling import rejection_sampling


class alphamu(object):
    def pdf(self, x, alpha, mu):
        return (alpha * mu ** mu * np.power(x, alpha * mu - 1.0)
                * np.exp(-mu * x ** alpha) / sps.gamma(mu))

    def rvs(self, low, high, alpha, mu, nsamples):
	return rejection_sampling(self.pdf, low, high, nsamples, alpha, mu)

class etamu(object):
    def pdf(self, x, eta, mu):
        return (4.0 * np.sqrt(np.pi) * mu ** (mu + 0.5) * x ** (2 * mu)
                * np.exp(-2.0 * mu * x * x / (1.0 + eta))
                * sps.ive(mu - 0.5, 2.0 * eta * mu * x * x / (1.0 - eta * eta))
                / (np.power(eta, mu - 0.5) * np.sqrt(1.0 - eta * eta)
                * sps.gamma(mu)))

    def rvs(self, low, high, eta, mu, nsamples):
        return rejection_sampling(self.pdf, low, high, nsamples, eta, mu)

class kappamu(object):
    def pdf(self, x, kappa, mu):
	return (2.0 * mu * (1.0 + kappa) ** ((mu + 1.0) / 2.0) * x ** mu
                * np.exp(-mu * (1 + kappa) * x * x - mu * kappa + 2 * x * mu
                * np.sqrt(kappa * (1 + kappa)))
                * sps.ive(mu - 1, 2 * mu * x * np.sqrt(kappa * (1.0 + kappa)))
                / (kappa ** ((mu-1.0)/2.0)))

    def kappa_mu_rvs(self, low, high, kappa, mu, nsamples):
	return rejection_sampling(self.pdf, low, high, nsamples, kappa, mu)
