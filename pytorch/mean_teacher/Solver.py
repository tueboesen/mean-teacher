from scipy.sparse import coo_matrix,csr_matrix, identity,triu,tril,diags,spdiags
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spsolve_triangular
import numpy as np


def gschmidt(V):
# Input: V is an m by n matrix of full rank m<=n
# Output: an m-by-n upper triangular matrix R
# and an m-by-m unitary matrix Q so that A = Q*R.
    nx , nd = V.shape
    R = np.zeros((nd,nd))
    Q = np.zeros((nx,nd))
    R[0,0] = np.linalg.norm(V[:,0])
    Q[:,0] = V[:,0] / R[0,0]
    for k in range(1, nd):
        R[0:k-1,k] = Q[:,0:k-1].T @ V[:,k]
        Q[:,k] = V[:,k] - Q[:,0:k-1] @ R[0:k-1,k]
        R[k,k] = np.linalg.norm(Q[:,k])
        Q[:,k] = Q[:,k] / R[k,k]
    return Q,R

def blkPCG(A,B,M,MaxIter=50,tol=1e-6,ortho=False):
    x = np.zeros(B.shape)
    R = B.copy()
    Z = M(R)
    if ortho:
        P, _ = gschmidt(Z)
    else:
        P = Z.copy()
    for i in range(MaxIter):
        # print("________________")
        # print(A.shape)
        # print(P.shape)
        # print("________________")
        Q = A @ P
        PTQ = P.T @ Q
        PTQinv = np.linalg.pinv(PTQ)
        alpha = PTQinv @ (P.T @ R)

        x += P @ alpha
        R -= Q @ alpha

        res = np.max(np.sqrt(np.sum(R * R,axis=0)) / np.sqrt(np.sum(B * B,axis=0)))
        if np.linalg.norm(res) < tol:
            break

        Z = M(R)
        beta = - PTQinv @ (Q.T @ Z)
        if ortho:
            P, _ = gschmidt(Z + P @ beta)
        else:
            P = Z + P * beta
    return x.T
