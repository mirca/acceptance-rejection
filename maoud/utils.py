import numpy as np
import scipy.stats as stats

def marcumq(x, df, nc):
    """Computes the Marcum-Q function.

    Parameters
    ----------
    x : input array
        The points where to evaluate the Marcum-Q function.
    df : int
        The degrees of freedom.
    nc : float
        The non-centrality parameter.

    Returns
    -------
    y : output array
        Array containing the values of the Marcum-Q function
        evaluated at `x`.
    """

    return stats.ncx2.sf(x*x, 2*df, nc*nc)

def mpsk(M, size):
    """Generates a random sample of symbols from a normalized M-PSK
    constellation.

    Parameters
    ----------
    M : int
        Order of the constellation.
    size : int
        Number of symbols to generate.
    """

    constellation = np.exp(1j * np.linspace(-np.pi / 4, 7 * np.pi / 4 - 2 * np.pi / M, M)) / np.sqrt(M)
    symbols = np.random.random_integers(0, M-1, size)
    return constellation[symbols]
