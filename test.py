import numpy as np
from euclidean_nmf import euclidean_nmf
from kullback_leibler_nmf import kullback_leibler_nmf

V = np.random.rand(10, 10)

e_nmf = euclidean_nmf(V=V, n_components=10)
e_nmf.nmf()

kl_nmf = kullback_leibler_nmf(V=V, n_components=10)
kl_nmf.nmf()