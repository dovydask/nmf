import numpy as np
from euclidean_nmf import euclidean_nmf
from kullback_leibler_nmf import kullback_leibler_nmf
from utils import kl_divergence

np.random.seed(1)
V = np.random.rand(10, 10)

e_nmf = euclidean_nmf(V=V)
e_nmf.nmf(n_components=10)

kl_nmf = kullback_leibler_nmf(V=V)
kl_nmf.nmf(n_components=10)