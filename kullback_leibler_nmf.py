import numpy as np
import math
from utils import kl_divergence

class kullback_leibler_nmf():
    """
    Standard non-negative matrix factorization (NMF) based on Lee & Seung NMF 
    algorithm. Uses Kullback-Leibler divergence with simple multiplicative 
    updates (slow).
    """

    def __init__(self, V, n_components, n_iters=100,
                 gradient=False, seed=None):
        self.V = V
        self.n_components = n_components
        self.n_iters = n_iters
        self.W = None
        self.H = None
        self.seed = seed
        self.gradient = gradient

    def nmf(self, verbose=True):
        """
        Apply non-negative matrix factorization (NMF) minimizing 
        Kullback-Leibler divergence to matrix V through multiplicative updates.
        Return matrices W and H (basis and coefficients of V). 
        """

        if verbose: 
            print("Kullback-Leibler NMF:")

        np.random.seed(self.seed)
        self.W = np.random.rand(self.V.shape[0], self.n_components)
        self.H = np.random.rand(self.n_components, self.V.shape[1])

        for it in range(self.n_iters):
            if self.gradient:
                print("Not yet!")
            else:
                self._update_H()
                self._update_W()

            if verbose and it % 10 == 0:
                print("Iteration", it, "divergence:", 
                      kl_divergence(self.V, self.W.dot(self.H)))

        if verbose: 
            print("End of factorization.\n")

    def _update_H(self):
        """
        Multiplicative update of H matrix for KL NMF.
        """

        for a in range(self.H.shape[0]):
            for m in range(self.H.shape[1]):
                p1 = 0
                p2 = 0
                for i in range(self.V.shape[0]):
                    p1 += self.W[i, a]*self.V[i, m]/(self.W.dot(self.H))[i, m]
                for k in range(self.W.shape[0]):
                    p2 += self.W[k, a]
                self.H[a, m] = self.H[a, m]*(p1/p2)

    def _update_W(self):
        """
        Multiplicative update of W matrix for KL NMF.
        """

        for i in range(self.W.shape[0]):
            for a in range(self.W.shape[1]):
                p1 = 0
                p2 = 0
                for m in range(self.H.shape[1]):
                    p1 += self.H[a, m]*self.V[i, m]/(self.W.dot(self.H))[i, m]
                for n in range(self.H.shape[1]):
                    p2 += self.H[a, n]
                self.W[i, a] = self.W[i, a]*(p1/p2)