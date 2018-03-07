from abc import ABC, abstractmethod
import numpy as np
import math
import scipy.special as sps
import scipy.stats as stats
from .sampling import rejection_sampling


class Distribution(ABC):
    """An abstract class for a probability distribution.
    """
    @abstractmethod
    def pdf(self, x):
        """Defines a 1D probability density function.

        Parameters
        ----------
        x : 1darray
            Points where to evaluate the pdf.

        Returns
        -------
        pdf : 1darray
            pdf values at `x`.
        """
        pass

    def rvs(self, x, size):
        """Returns a sample of length `size` in the range
        [x.min(), x.max()].

        Parameters
        ----------
        x : array
            The support of the pdf.
        size : int
            Size of the sample (number of realizations).
        """
        return rejection_sampling(self.pdf, x, size)


class ComplexDistribution(Distribution):
    """Defines a class for probability distribution
    arising from a complex random variable Z = X + jY,
    where j = sqrt(-1).
    """
    def rvs(self, x, y, size):
        """Returns a sample of length `size` in the range
        [x.min(), x.max()] and [y.min(), y.max()] for the
        real and imaginary parts.

        Parameters
        ----------
        x, y : arrays
            The support of the pdf of the real and imaginary parts,
            respectively.
        size : int
            Size of the sample (number of realizations).
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")

        return (rejection_sampling(self.pdf, x, size)
                + 1j * rejection_sampling(self.pdf, y, size))


class AlphaMu(Distribution):
    """Defines the α — μ probability distribution.
    For the theorectical aspects of the α — μ distribution, see
    M. D. Yacoub, The α — μ distribution: A physical fading model for the Stacy
    distribution, IEEE Trans. Veh. Technol., vol. 56, no. 1, pp. 27–34, 2007.

    Attributes
    ----------
    alpha, mu : float, float
        Parameters that define the α — μ distribution.
    """
    def __init__(self, alpha, mu):
        self.alpha = alpha
        self.mu = mu

    def pdf(self, x):
        """Defines a univariate α — μ  probability density function.

        Parameters
        ----------
        x : 1darray
            Points where to evaluate the pdf.

        Returns
        -------
        pdf : 1darray
            pdf values at `x`.
        """
        return (self.alpha * self.mu ** self.mu * x ** (self.alpha * self.mu - 1.0)
                * np.exp(-self.mu * x ** self.alpha) / sps.gamma(self.mu))


class EtaMu(Distribution):
    """Defines the \eta — μ probability distribution.
    For the theorectical aspects of the \eta — μ distribution, see
    [ADD REFERENCE]

    Attributes
    ----------
    eta, mu : float, float
        Parameters that define the eta — μ distribution.
    """

    def __init__(self, eta, mu):
        self.eta = eta
        self.mu = mu

    def pdf(self, x):
        return (4.0 * np.sqrt(np.pi) * self.mu ** (self.mu + 0.5) * x ** (2 * self.mu)
                * np.exp(-2. * self.mu * x * x / (1. + self.eta))
                * sps.ive(self.mu - 0.5, 2. * self.eta * self.mu * x * x / (1. - self.eta * self.eta))
                / (self.eta ** (self.mu - 0.5) * np.sqrt(1. - self.eta * self.eta)
                * sps.gamma(self.mu)))


class KappaMu(Distribution):
    """Defines the \kappa — μ probability distribution.
    For the theorectical aspects of the \kappa — μ distribution, see
    [ADD REFERENCE]

    Attributes
    ----------
    kappa, mu : float, float
        Parameters that define the kappa — μ distribution.
    """

    def __init__(self, kappa, mu):
        self.kappa = kappa
        self.mu = mu

    def pdf(self, x):
        return (2. * self.mu * (1. + self.kappa) ** ((self.mu + 1.) * .5)
                * np.power(x, self.mu)
                * np.exp(-self.mu * (1. + self.kappa) * x * x - self.mu * self.kappa
                         + 2. * x * self.mu * math.sqrt(self.kappa * (1. + self.kappa)))
                * sps.ive(self.mu - 1., 2. * self.mu * x * math.sqrt(self.kappa * (1. + self.kappa)))
                / self.kappa ** ((self.mu - 1.) * .5))


class ComplexAlphaMu(ComplexDistribution):
    def __init__(self, alpha, mu):
        self.alpha = alpha
        self.mu = mu

    def pdf(self, x):
        return (self.mu ** (self.mu * 0.5) * np.abs(x) ** (self.mu - 1.0) * np.exp(-self.mu * x * x)
                / sps.gamma(self.mu * 0.5))

class ComplexEtaMu(ComplexDistribution):
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

    def real_part(self, x):
        p = np.sqrt(self.kappa / (1.0 + self.kappa)) * np.cos(self.phi)
        sigma2 = 1.0 / (2.0 * self.mu * (1.0 + self.kappa))

        return (np.abs(x) ** (0.5 * self.mu) * np.exp(-(x - p) ** 2 / (2.0 * sigma2))
                * sps.iv(self.mu * 0.5 - 1.0, np.abs(p * x) / sigma2)
                / (2.0 * sigma2 * np.abs(p) ** (0.5 * self.mu - 1.0) * np.cosh(p * x / sigma2)))

    def imag_part(self, x):
        q = np.sqrt(self.kappa / (1.0 + self.kappa)) * np.sin(self.phi)
        sigma2 = 1.0 / (2.0 * self.mu * (1.0 + self.kappa))

        return (np.abs(x) ** (0.5 * self.mu) * np.exp(-(x - q) ** 2 / (2.0 * sigma2))
                * sps.iv(self.mu * 0.5 - 1.0, np.abs(q * x) / sigma2)
                / (2.0 * sigma2 * np.abs(q) ** (0.5 * self.mu - 1.0) * np.cosh(q * x / sigma2)))

    def rvs(self, x, y, size):
        return (rejection_sampling(self.real_part, x, y, size)
                + 1j * rejection_sampling(self.imag_part, x, y, size))
