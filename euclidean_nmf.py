import numpy as np
from utils import euclidean_distance

class euclidean_nmf():
    """
    Standard non-negative matrix factorization (NMF) based on
    Lee & Seung NMF algorithm. Uses euclidean distance with
    simple multiplicative updates.
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
        Apply non-negative matrix factorization (NMF)
        minimizing Euclidean norm to matrix V through multiplicative
        updates. Return matrices W and H (basis and coefficients of V). 
        """

        if verbose: 
            print("Euclidean NMF:")

        np.random.seed(self.seed)
        self.W = np.random.rand(self.V.shape[0], self.n_components)
        self.H = np.random.rand(self.n_components, self.V.shape[1])

        for it in range(self.n_iters):
            if self.gradient:
                print("Not yet!")
            else:
                self._update_mult()
            
            if verbose and it % 10 == 0:
                print("Iteration", it, "distance:", 
                      euclidean_distance(self.V, self.W.dot(self.H)))

        if verbose:
            print("End of factorization.\n")

    def _update_mult(self):
        """
        Multiplicative update of W and H matrices for Euclidean NMF.
        """

        self.H = self.H*((self.W.T.dot(self.V))/(self.W.T.dot(self.W).dot(self.H)))
        self.W = self.W*((self.V.dot(self.H.T))/(self.W.dot(self.H).dot(self.H.T)))