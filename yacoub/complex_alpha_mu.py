import numpy as np
import scipy.special as sps
from rejection_sampling import rejection_sampling

def complex_alpha_mu_pdf(x,alpha,mu):
	# Returns the pdf of X (or Y), assuming that R = X + jY is complex Nakagami distributed
	return np.power(mu, mu * 0.5) * np.power(np.abs(x), mu - 1.0) * np.exp(-mu * x * x) / sps.gamma(mu*0.5)

def complex_alpha_mu_rvs(K,alpha,mu):
	return (rejection_sampling(complex_alpha_mu_pdf, -10, 10, K, alpha, mu)
                + 1j * rejection_sampling(complex_alpha_mu_pdf, -10, 10, K, alpha, mu))
