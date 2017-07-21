import numpy as np
import scipy.special as sps
from rejection_sampling import rejection_sampling

def complex_eta_mu_pdf(x, eta, mu):
	# Returns the pdf of X, assuming that R = X + jY is complex eta-mu distributed
	return (np.power(2.0 * mu, mu) * np.power(np.abs(x), 2.0 * mu - 1.0)
                * np.exp(-2.0 * mu * x * x/ (1.0 - eta))
                / (sps.gamma(mu) * np.power(1.0 - eta, mu)))

def complex_eta_mu_rvs(K, eta, mu):
	return (rejection_sampling(complex_eta_mu_pdf, -30, 30, K, eta, mu)
                + 1j * rejection_sampling(complex_eta_mu_pdf, -30, 30, K, eta, mu))
