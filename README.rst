**maoud: simulation of generalized fading channels in Python**

Let's say you have a complicated density function for which there is no implementation in ``Scipy`` e.g., Yacoub's Kappa-Mu:

```
import numpy as np
import scipy.special as sps

def kappa_mu_pdf(x, kappa, mu):
    return (2.0 * mu * np.power(1.0 + kappa, (mu + 1.0) / 2.0) * np.power(x, mu)
            * np.exp(-mu * (1.0 + kappa) * x * x - mu * kappa + 2 * x * mu
            * np.sqrt(kappa * (1.0 + kappa))) * sps.ive(mu - 1, 2 * mu * x
            * np.sqrt(kappa * (1.0 + kappa))) / (np.power(kappa, (mu - 1.0) / 2.0)))
```

Then you might want to do the following in order to draw samples

```
from sampling import rejection_sampling

n_samples = int(1e6)
kappa = 0.75
mu = 0.87
low = 1e-6
high = 3

kappa_mu_samples = rejection_sampling(kappa_mu_pdf, low, high,
                                      n_samples, kappa, mu)
```

To verify that the samples are in accordance with Yacoub's Kappa-Mu density, let's plot the histogram of the samples:

```
import matplotlib.pyplot as plt

x = np.linspace(1e-6, 3, 1000)
y = kappa_mu_pdf(x, kappa, mu)

plt.plot(x, y)
plt.hist(kappa_mu_samples, bins=50, normed=True)
```

![sick plot](https://github.com/mirca/acceptance-rejection/raw/master/kappa_mu.png)

SHAZAM!!

**Citation**

If you made use of the code available in this repository, please consider
citing the following work:

**On the performance of the energy detector subject to impulsive noise in κ—μ, α—μ, and η—μ fading channels**
JVM Cardoso, WJL Queiroz, H Liu, MS Alencar - *Tsinghua Science and Technology*, 2017.

```
@ARTICLE{7986939,
author={J. V. M. Cardoso and W. J. L. Queiroz and H. Liu and M. S. Alencar},
journal={Tsinghua Science and Technology},
title={On the performance of the energy detector subject to impulsive noise in #x03BA; #x2014; #x03BC;, #x03B1; #x2014; #x03BC;, and #x03B7; #x2014; #x03BC; fading channels},
year={2017},
volume={22},
number={4},
pages={360-367},
keywords={Detectors;Monte Carlo methods;Probability density function;Rayleigh channels;Signal to noise ratio;energy detector;fading channel;impulsive noise;spectrum sensing},
doi={10.23919/TST.2017.7986939},
ISSN={},
month={Aug},}
```
