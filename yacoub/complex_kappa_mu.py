import numpy as np
import scipy.special as sps
from rejection_sampling import rejection_sampling

def real_kappa_mu_pdf(x,kappa,mu,phi):
	# Returns the pdf of X, assuming that R = X + jY is complex kappa-mu distributed
	p = np.sqrt(kappa/(1.0+kappa))*np.cos(phi)
	sigma2 = 1.0/(2.0*mu*(1.0+kappa))

	return np.power(np.abs(x),0.5*mu)*np.exp(-(x-p)*(x-p)/(2.0*sigma2))*sps.iv(mu*0.5-1.0,np.abs(p*x)/sigma2)/(2.0*sigma2*np.power(np.abs(p),0.5*mu-1.0)*np.cosh(p*x/sigma2))

def imag_kappa_mu_pdf(x,kappa,mu,phi):
	q = np.sqrt(kappa/(1.0+kappa))*np.sin(phi)
	sigma2 = 1.0/(2.0*mu*(1.0+kappa))

	return np.power(np.abs(x),0.5*mu)*np.exp(-(x-q)*(x-q)/(2.0*sigma2))*sps.iv(mu*0.5-1.0,np.abs(q*x)/sigma2)/(2.0*sigma2*np.power(np.abs(q),0.5*mu-1.0)*np.cosh(q*x/sigma2))

def complex_kappa_mu_rvs(K,kappa,mu,phi):
	return rejection_sampling(real_kappa_mu_pdf,-10,10,K,kappa,mu,phi) + 1j*rejection_sampling(imag_kappa_mu_pdf,-10,10,K,kappa,mu,phi)
