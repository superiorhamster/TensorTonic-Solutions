import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    # return np.array(A).T
    N,M = np.array(A).shape
    B = np.zeros((M,N))
    for i in range(N):
        for j in range(M):
            B[j][i] = A[i][j]

    return B
    pass
