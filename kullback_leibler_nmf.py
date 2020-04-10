import numpy as np
import math
from utils import kl_divergence

class kullback_leibler_nmf():
    """
    Standard non-negative matrix factorization (NMF) based on Lee & Seung NMF 
    algorithm. Uses Kullback-Leibler divergence with simple multiplicative 
    updates.

    Reference:
    J.J. Burred (2014). 
    Detailed derivation of multiplicative update rules for NMF.
    https://www.jjburred.com/research/pdf/jjburred_nmf_updates.pdf
    """

    def __init__(self, V):
        self.V = V
        self.W = None
        self.H = None

    def nmf(self, n_components, n_iters=100, gradient=False, seed=None, 
            verbose=True):
        """
        Apply non-negative matrix factorization (NMF) minimizing 
        Kullback-Leibler divergence to matrix V through multiplicative updates.
        Return matrices W and H (basis and coefficients of V). 
        """

        if verbose: 
            print("Kullback-Leibler NMF,", n_components, "components")

        np.random.seed(seed)
        self.W = np.random.rand(self.V.shape[0], n_components)

        np.random.seed(seed)
        self.H = np.random.rand(n_components, self.V.shape[1])

        for it in range(n_iters):
            if gradient:
                print("Not yet!")
            else:
                self._update()

            if verbose and it % 10 == 0:
                print("Iteration", it, "divergence:", 
                      kl_divergence(self.V, self.W.dot(self.H)))

        if verbose: 
            print("End of factorization.")
            print("Final divergence:", 
                  kl_divergence(self.V, self.W.dot(self.H)), "\n")

    def _update(self):
        """
        Multiplicative update of W and H matrices for Kullback-Leibler NMF.
        """

        ones_H = np.ones((self.H.shape[0], self.H.shape[1]))
        ones_W = np.ones((self.W.shape[0], self.W.shape[1]))
        self.H = self.H*(self.W.T.dot((self.V/self.W.dot(self.H)))/(self.W.T.dot(ones_W)))
        self.W = self.W*((self.V/self.W.dot(self.H)).dot(self.H.T)/(ones_H.dot(self.H.T)))