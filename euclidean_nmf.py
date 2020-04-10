import numpy as np
from utils import euclidean_distance

class euclidean_nmf():
    """
    Standard non-negative matrix factorization (NMF) based on
    Lee & Seung NMF algorithm. Uses euclidean distance with
    simple multiplicative updates.

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
        Apply non-negative matrix factorization (NMF)
        minimizing Euclidean norm to matrix V through multiplicative
        updates. Return matrices W and H (basis and coefficients of V). 
        """

        if verbose: 
            print("Euclidean NMF, ", n_components, "components")

        np.random.seed(seed)
        self.W = np.random.rand(self.V.shape[0], n_components)

        np.random.seed(seed)
        self.H = np.random.rand(n_components, self.V.shape[1])

        for it in range(n_iters):
            if gradient:
                print("Not yet!")
            else:
                self._update_mult()
            
            if verbose and it % 10 == 0:
                print("Iteration", it, "distance:", 
                      euclidean_distance(self.V, self.W.dot(self.H)))

        if verbose:
            print("End of factorization.")
            print("Final distance:", 
                  euclidean_distance(self.V, self.W.dot(self.H)), "\n")

    def _update_mult(self):
        """
        Multiplicative update of W and H matrices for Euclidean NMF.
        """
        
        self.H = self.H*((self.W.T.dot(self.V))/(self.W.T.dot(self.W).dot(self.H)))
        self.W = self.W*((self.V.dot(self.H.T))/(self.W.dot(self.H).dot(self.H.T)))