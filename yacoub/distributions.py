from abc import ABC, abstractmethod
import numpy as np
import scipy.special as sps
import scipy.stats as stats
from sampling import rejection_sampling


class Distribution(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    def rvs(self, low, high, nsamples):
        return rejection_samples(self.pdf, low, high, nsamples)


class ComplexDistributions(Distribution):
    def rvs(self, low, high, nsamples):
        return (rejection_samples(self.pdf, low, high, nsamples)
                + 1j * rejection_samples(self.pdf, low, high, nsamples))


class AlphaMu(Distribution):
    def __init__(self, alpha, mu):
        self.alpha = alpha
        self.mu = mu

    def pdf(self, x):
        return (self.alpha * self.mu ** self.mu * x ** (self.alpha * self.mu - 1.0)
                * np.exp(-self.mu * x ** self.alpha) / sps.gamma(self.mu))


class EtaMu(Distribution):
    def __init__(self, eta, mu):
        self.eta = eta
        self.mu = mu

    def pdf(self, x):
        return (4.0 * np.sqrt(np.pi) * self.mu ** (self.mu + 0.5) * x ** (2 * self.mu)
                * np.exp(-2.0 * self.mu * x * x / (1.0 + self.eta))
                * sps.ive(self.mu - 0.5, 2.0 * self.eta * self.mu * x * x / (1.0 - self.eta * self.eta))
                / (self.eta ** (self.mu - 0.5) * np.sqrt(1.0 - self.eta * self.eta)
                * sps.gamma(self.mu)))


class KappaMu(Distribution):
    def __init__(self, kappa, mu):
        self.kappa = kappa
        self.mu = mu

    def pdf(self, x):
	return (2.0 * self.mu * (1.0 + self.kappa) ** ((self.mu + 1.0) / 2.0) * x ** self.mu
                * np.exp(-self.mu * (1 + self.kappa) * x * x - self.mu * self.kappa + 2 * x * self.mu
                * np.sqrt(self.kappa * (1 + self.kappa)))
                * sps.ive(self.mu - 1, 2 * self.mu * x * np.sqrt(self.kappa * (1.0 + self.kappa)))
                / (self.kappa ** ((self.mu - 1.0) / 2.0)))


class ComplexAlphaMu(ComplexDistribution):
    def __init__(self, alpha, mu):
        self.alpha = alpha
        self.mu = mu

    def pdf(self, x):
	return (self.mu ** (self.mu * 0.5) * np.abs(x) ** (self.mu - 1.0) * np.exp(-self.mu * x * x)
                / sps.gamma(self.mu * 0.5))


class ComplexEtaMu(ComplexDistributon):
    def __init__(self, eta, mu):
        self.eta = eta
        self.mu = mu

    def pdf(self, x):
        return ((2.0 * self.mu) ** self.mu * np.abs(x) ** (2.0 * self.mu - 1.0)
                * np.exp(-2.0 * self.mu * x * x/ (1.0 - self.eta))
                / (sps.gamma(self.mu) * np.power(1.0 - self.eta, self.mu)))


class ComplexKappaMu(object):
    def __init__(self, kappa, mu, phi):
        self.kappa = kappa
        self.mu = mu
        self.phi = phi

    def real_pdf(self, x):
        p = np.sqrt(self.kappa / (1.0 + self.kappa)) * np.cos(self.phi)
	sigma2 = 1.0 / (2.0 * self.mu * (1.0 + self.kappa))

	return (np.abs(x) ** (0.5 * self.mu) * np.exp(-(x - p) ** 2 / (2.0 * sigma2))
                * sps.iv(self.mu * 0.5 - 1.0, np.abs(p * x) / sigma2)
                / (2.0 * sigma2 * np.abs(p) ** (0.5 * self.mu - 1.0) * np.cosh(p * x / sigma2)))

    def imag_pdf(self, x):
	q = np.sqrt(self.kappa / (1.0 + self.kappa)) * np.sin(self.phi)
	sigma2 = 1.0 / (2.0 * self.mu * (1.0 + self.kappa))

	return (np.abs(x) ** (0.5 * self.mu) * np.exp(-(x - q) ** 2 / (2.0 * sigma2))
                * sps.iv(self.mu * 0.5 - 1.0, np.abs(q * x) / sigma2)
                / (2.0 * sigma2 * np.abs(q) ** (0.5 * self.mu - 1.0) * np.cosh(q * x / sigma2)))

    def rvs(self, low, high, nsamples):
        return (rejection_samples(self.real_pdf, low, high, nsamples)
                + 1j * rejection_samples(self.imag_pdf, low, high, nsamples))
