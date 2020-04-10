import numpy as np
import math

def euclidean_distance(A, B):
    """
    Euclidean norm of matrices A and B
        
    Input:
        A -- m x n matrix
        B -- m x n matrix

    Output:
        dist -- Squared Euclidean distance (norm)
    """

    dist = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            dist += (A[i, j] - B[i, j])**2
    
    return dist

def kl_divergence(A, B):
    """
    Kullback-Leibler divergence of matrices A and B

    Input:
        A -- m x n matrix
        B -- m x n matrix

    Output:
        cost -- Kullback-Leibler divergence
    """

    assert A.shape == B.shape
    div = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            div += (A[i, j]*math.log(A[i, j]/B[i, j]) - A[i, j] + B[i, j])
    return div