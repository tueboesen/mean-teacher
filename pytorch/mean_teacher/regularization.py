from annoy import AnnoyIndex
import time
import scipy.io as sio
from os import walk
import h5py
import numpy as np
import datetime
import statistics
from scipy.sparse import coo_matrix,csr_matrix, identity,triu,tril,diags,spdiags
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spsolve_triangular
import torch
import hnswlib

from mean_teacher.Solver import blkPCG


def ANN_hnsw(x,k=10):
    nsamples = len(x)
    dim = len(x[0])
    print('dimension=', dim)
    print('nsamples =', nsamples)
    t1 = time.time()
    # Generating sample data
    data = x
    data_labels = np.arange(nsamples)

    # Declaring index
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=nsamples, ef_construction=200, M=32)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(100)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k)
    t2 = time.time()
    Js = []
    Is = []
    for i,subnn in enumerate(labels):
        for itemnn in subnn:
            Js.append(itemnn)
            Is.append(i)
    Vs = np.ones_like(Js)
    A = csr_matrix((Vs, (Is,Js)),shape=(nsamples,nsamples))
    A = (A + A.T).sign()
    t3 = time.time()
    print('Time spent finding ANN   : {}'.format(t2 - t1))
    print('Time spent building A    : {}'.format(t3 - t2))
    return A,np.mean(distances)


def ANN_annoy(x,k=10):
    acc_factor = 30
    ntrees = 300
    nsamples = len(x)
    dim = len(x[0])
    print('dimension=', dim)
    print('nsamples =', nsamples)
    search_k = int(ntrees * k * acc_factor)
    t = AnnoyIndex(dim, "angular")  # Length of item vector that will be indexed
    # Now we add some data
    for i in range(nsamples):
        t.add_item(i, x[i])
    t1 = time.time()
    t.build(ntrees)  # Now we build the trees
    t2 = time.time()
    # Now lets find the actual neighbors
    neighbors = []
    distances = []
    for i in range(nsamples):
        idx, dist = t.get_nns_by_item(i, k, include_distances=True)
        neighbors.append(idx)
        distances.append(dist)
    t3 = time.time()
    dist = [item for sublist in distances for item in sublist]
    Js = []
    Is = []
    # Vs = []
    # ctn = 0
    # for (idx,vals) in enumerate(distances):
    #     if vals[0] > 0:
    #         print(idx)
    #         ctn += 1


    # for i,(subnn, subdist) in enumerate(zip(neighbors, distances)):
    #     for (itemnn,itemdist) in zip(subnn, subdist):
    #         Js.append(itemnn)
    #         Is.append(i)
    #         Vs.append(1-itemdist)
    Vs = np.ones_like(Js)
    A = csr_matrix((Vs, (Is,Js)),shape=(nsamples,nsamples))
    A = (A + A.T).sign()
    t4 = time.time()
    print('Time spent building trees: {}'.format(t2 - t1))
    print('Time spent finding knns  : {}'.format(t3 - t2))
    print('Time spent building A    : {}'.format(t4 - t3))
    print('Total time spent         : {}'.format(t4 - t1))
    return A,statistics.median(dist)


def GraphLaplacian(X,A,dist):
        t1 = time.time()
        if isinstance(X, torch.Tensor):
            X=X.numpy()
        A=A.tocoo()
        n,_=A.shape
        I = A.row
        J = A.col
        t2 = time.time()
        tmp = np.sum((X[I] - X[J]) ** 2, axis=1)
        t3 = time.time()
        V = np.exp(-tmp/dist)
        W = coo_matrix((V, (I, J)), shape=(n, n))
        D = coo_matrix((n, n))
        coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
        Dh = np.sqrt(D)
        np.reciprocal(Dh.data, out=Dh.data)
        t4 = time.time()
        L = D - W
        L = Dh @ L @ Dh
        LT = L.transpose()
        L = 0.5 * (LT + L)
        t5 = time.time()
        print("L {}".format(t2 - t1))
        print("L {}".format(t3 - t2))
        print("L {}".format(t4 - t3))
        print("L {}".format(t5 - t4))
        # tmp = np.empty_like(J,dtype=X.dtype)
        # for idx, (ii, jj) in enumerate(zip(I, J)):
        #     tmp[idx] = sum((X[ii] - X[jj])**2)


        return L


def ANN_W(X, A,alpha):
    t1 = time.time()
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocoo()
    nx, _ = A.shape
    Is = A.row
    Js = A.col

    # for (i,j) in zip(I,J):
    V = np.sum((X[Is] * X[Js]) ** 3, axis=1)
    print("Number of V elements less than zero {}".format(np.sum(V < 0)))
    print("smallest V {}".format(np.min(V)))
    assert np.min(V) > 0, print("some elements of V are less than zero")

    Aa = coo_matrix((V, (Is, Js)), shape=(nx, nx))
    W = Aa.T + Aa
    D = coo_matrix((nx, nx))
    coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
    Dh = np.sqrt(D)
    np.reciprocal(Dh.data, out=Dh.data)
    Ww = Dh @ W @ Dh
    I = identity(nx)
    L = (I - alpha * Ww)
    t2 = time.time()
    print("ANN_W {}".format(t2 - t1))
    return L

def ANN_W2(X, A,alpha):
    t1 = time.time()
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocoo()
    nx, _ = A.shape
    Is = A.row
    Js = A.col

    # for (i,j) in zip(I,J):
    V = 1-np.sum((X[Is] * X[Js]), axis=1)/(np.sqrt(np.sum(X[Is]**2, axis=1))*np.sqrt(np.sum(X[Js]**2, axis=1)))
    assert np.min(V) > 0, print("some elements of V are less than zero")
    assert np.max(V) < 1, print("some elements of V are larger than 1")

    Aa = coo_matrix((V, (Is, Js)), shape=(nx, nx))
    W = Aa.T + Aa
    D = coo_matrix((nx, nx))
    coo_matrix.setdiag(D, np.squeeze(np.array(np.sum(W, axis=0))))
    Dh = np.sqrt(D)
    np.reciprocal(Dh.data, out=Dh.data)
    Ww = Dh @ W @ Dh
    I = identity(nx)
    L = (I - alpha * Ww)
    t2 = time.time()
    print("ANN_W {}".format(t2 - t1))
    return L



def softmaxloss(U,C):
    _, nx = C.shape
    S = np.exp(U)
    F = - np.sum(np.sum(C*U)) + np.sum(np.log(np.sum(S, axis=0)))
    F = F / nx
    dF = (-C + S ) / np.sum(S, axis=0) / nx
    return F, dF

def SSL_ADMM(U,idx,C,L,alpha,beta,rho,lambd,V,maxIter):
    print("starting SSL_ADMM")
    print("U=",U.shape)
    print("idx=",len(idx))
    print("C=",C.shape)
    print("L=",L.shape)
    print("L nnz=",L.nnz)
    print("type L =",type(L))
    print("V=",V.shape)
    print("lambd=",lambd.shape)
    t1 = time.time()
    L = L.tocsr()
    print("type L (recast to csr) =",type(L))
    nl = len(idx)
    nc, nx = U.shape
    _, nk = C.shape
    # M(x) = tril(A)\(diag(A).*(triu(A)\x));
    A = L + beta * identity(nx)
    print("type A =",type(L))
    def precond(x):
        return spsolve(tril(A, format='csc'), (A.diagonal() * spsolve(triu(A, format='csc'), x, permc_spec='NATURAL')), permc_spec='NATURAL')
    # def precond(x):
    #     return spsolve_triangular(tril(A, format='csr'), (A.diagonal() * spsolve_triangular(triu(A, format='csr'), x, lower=False)))
    M = LinearOperator(matvec=precond, shape=(nx, nx), dtype=float)
    Is=np.arange(0, nl)
    Js =idx
    Vs =np.ones_like(Is)
    P = coo_matrix((Vs, (Is, Js)), shape=(nl, nx))
    print("Iter    Loss       Reg       Misfit    Lagrange      df          dU         muls \n")
    Iter = 1
    muls = 1
    t2 = time.time()
    print("Time on init {}".format(t2-t1))
    dU = np.empty_like(U)
    tol = 1e-5
    maxiter = 100
    while True:
        t3 = time.time()

        F, dF = softmaxloss(U @ P.T, C)
        dF = dF @ P
        t4 = time.time()
        reg = 0.5*alpha*np.trace(U @ L @ U.T)/nx
        misfit = F
        lagrange = rho /2 * np.linalg.norm(U.T - V.T + lambd.T)**2
        f = reg + misfit + lagrange
        assert f > 0, "Error: Negative loss function"
        df =  alpha * U @ L / nx + dF + rho * (U - V + lambd)
        t5 = time.time()
        dU=blkPCG(A,df.T,M,MaxIter=maxiter,tol=tol,ortho=True)
        # for i in range(nc):
        #     dU[i, :], _ = cg(A, df[i, :], M=M,tol=tol,maxiter=maxiter)
        t6 = time.time()
        IsIter = 1
        while True:
            Utry = U - muls * dU
            Ftry,_ = softmaxloss(Utry @ P.T, C)
            ftry = 0.5*alpha*np.trace(Utry @ L @ Utry.T)/nx + Ftry + rho /2 * np.linalg.norm(Utry.T - V.T + lambd.T)**2
            if ftry < f:
                if IsIter == 1:
                    muls *= 1.3
                break
            else:
                muls = muls/2
                IsIter += 1
                assert IsIter < 20, "Line search failed"
        t7 = time.time()
        U = Utry
        cp = np.exp(U) / np.sum(np.exp(U),axis=0)
        print("SoftMax {}".format(t4 - t3))
        print("f and df {}".format(t5 - t4))
        print("solver {}".format(t6 - t5))
        print("linesearch {}".format(t7 - t6))
        print("{:3d} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2f} \n".format(Iter,f,reg,misfit,lagrange,muls))
        Iter += 1
        if Iter > maxIter:
            break
    return U, cp

def SSL_ADMM_dummy(U,idx,C,L,alpha,beta,rho,lambd,V,maxIter):
    nc, nx = U.shape
    cp = np.random.rand(nc,nx)
    return U, cp

def SSL_Icen(L,Y):
    nx,nc = Y.shape
    cp = np.empty_like(Y)
    tol = 1e-6
    maxiter = 100
    t1 = time.time()
    for i in range(nc):
        cp[:,i], _ = cg(L, Y[:,i], tol=tol,maxiter=maxiter)
    t2 = time.time()
    print("CG {}".format(t2 - t1))
    return cp.T
#     cp = blkPCG(L, Y, M, MaxIter=maxiter, tol=tol, ortho=True)