from __future__ import division
import numpy as np
import copy

def Least_Trace(sc, X, Y, rho1, init = 2, rho_L2 = 0, tFlag = 1, maxIter=1000, tol=1e-4):

    task_num=len(X)
    dimension=X[0].shape[1]
    funcVal=[]
    XY=[]

    # precomputation.
    for t_idx in range(0,task_num):
        X[t_idx] = X[t_idx].T
        # XY.append( X[t_idx] * Y[t_idx] )

    #X_RDD = sc.parallelize(X).zipWithIndex().map(lambda (k,v) : (v,k))
    #Y_RDD = sc.parallelize(Y).zipWithIndex().map(lambda (k,v) : (v,k))
    #XY_RDD = X_RDD.join(Y_RDD).map( lambda (k,v) : (k, v[0]*v[1]) )

    #W0_prep = np.concatenate( XY_RDD.map( lambda (k,v) : v ).collect(), axis=1)
    
    X_RDD = sc.parallelize(X)
    Y_RDD = sc.parallelize(Y)
    XY_RDD = X_RDD.zip(Y_RDD).map( lambda (k,v) : (k*v) )
    W0_prep = np.concatenate( XY_RDD.collect(), axis=1)

    # initialize a starting point
    if init == 2:
        W0=np.zeros_like(W0_prep)
    else:
        # if opts.init == 0:
            W0=W0_prep
        # else:
        #     if isfield(opts,char('W0')):
        #         W0=opts.W0
        #         if (nnz(size(W0) - cat(dimension,task_num))):
        #             error(char('\n Check the input .W0'))
        #     else:
        #         W0=copy(W0_prep)

    bFlag=0 # this flag tests whether the gradient step only changes a little

    Wz = copy.copy(W0)
    Wz_old = copy.copy(W0)

    t = 1
    t_old = 0


    iter_ = 0
    gamma = 1
    gamma_inc = 2

    while (iter_ < maxIter):

        alpha = (t_old - 1) /t
        Ws = (1 + alpha) * Wz - alpha * Wz_old
        
        # compute function value and gradients of the search point
        # use map for function calls
        gWs  = gradVal_eval(sc, Ws,X_RDD,XY_RDD,rho_L2)
        Fs   = funVal_eval(sc, Ws,X_RDD,Y_RDD,rho_L2)
        
        while 1:
            Wzp,Wzp_tn=trace_projection(Ws - gWs / gamma,2 * rho1 / gamma)
            Fzp = funVal_eval  (sc,Wzp,X_RDD,Y_RDD,rho_L2)
            
            delta_Wzp = Wzp - Ws
            r_sum=np.linalg.norm(delta_Wzp,'fro') ** 2
            Fzp_gamma=Fs + np.sum(np.multiply(delta_Wzp, gWs)) + gamma / 2 * np.linalg.norm(delta_Wzp,'fro') ** 2
            
            if (r_sum <=1e-20):
                bFlag=1 # this shows that, the gradient step makes little improvement
                break
            
            if (Fzp <= Fzp_gamma):
                break
            else:
                gamma = gamma * gamma_inc;
        
        Wz_old=copy.copy(Wz)
        Wz=copy.copy(Wzp)

        funcVal.append(Fzp + rho1 * Wzp_tn)
        
        if (bFlag):
            # fprintf('\n The program terminates as the gradient step changes the solution very small.');
            break
        
        # test stop condition.
        if 0 == (tFlag):
            if iter_ >= 2:
                if (abs(funcVal[-1] - funcVal[-2]) <= tol):
                    break
        else:
            if 1 == (tFlag):
                if iter_ >= 2:
                    if (abs(funcVal[-1] - funcVal[-2]) <= tol * funcVal[-2]):
                        break
            else:
                if 2 == (tFlag):
                    if (funcVal[-1] <= tol):
                        break
                else:
                    if 3 == (tFlag):
                        if iter_ >= maxIter:
                            break
        iter_=iter_ + 1
        t_old=copy.copy(t)
        t=0.5 * (1 + (1 + 4 * t ** 2) ** 0.5)

    W = Wzp

    return W, funcVal



# get rid of parfor

def gradVal_eval(sc,W,X,XY,rho_L2):
    task_num = W.shape[1]
    #W_RDD = sc.parallelize(np.hsplit(W,task_num)).zipWithIndex().map(lambda (k,v) : (v,k))
    #XTXW_RDD = X.join(W_RDD).map( lambda (k,v) : (k, v[0]*(v[0].T*v[1]) ) )
    #grad_W = np.concatenate( XTXW_RDD.join(XY).map( lambda (k,v) : (k, v[0]-v[1]) ).sortByKey().map( lambda (k,v) : v ).collect() , axis =1)
    W_RDD = sc.parallelize(np.hsplit(W,task_num))
    XTXW_RDD = X.zip(W_RDD).map( lambda (k,v) : k*(k.T*v) )
    grad_W = np.concatenate( XTXW_RDD.zip(XY).map( lambda (k,v) : (k-v) ).collect(), axis=1 )
    grad_W = grad_W + rho_L2 * 2 * W
    return grad_W


def funVal_eval(sc,W,X,Y,rho_L2):
    task_num = W.shape[1]
    #W_RDD = sc.parallelize(np.hsplit(W,task_num)).zipWithIndex().map(lambda (k,v) : (v,k))
    #V_RDD = X.join(W_RDD).map( lambda (k,v) : (k, v[0].T*v[1]) )
    #V_RDD = V_RDD.join(Y).map( lambda (k,v) : (k, 0.5*np.linalg.norm(v[1]-v[0])**2) ).sortByKey().map( lambda (k,v) : v )
    W_RDD = sc.parallelize(np.hsplit(W,task_num))
    V_RDD = X.zip(W_RDD).map( lambda (k,v) : (k.T*v) )
    V_RDD = V_RDD.zip(Y).map( lambda (k,v) : (0.5*(np.linalg.norm(v-k)**2)) )
    funcVal = sum(V_RDD.collect())
    funcVal=funcVal + rho_L2 * np.linalg.norm(W,'fro') ** 2
    return funcVal

def trace_projection(L, alpha):
    d1,d2=L.shape
    if (d1 > d2):
        U,S,V=np.linalg.svd(L,0)
        thresholded_value=S - alpha / 2
        diag_S=thresholded_value*(thresholded_value > 0)
        L_hat=U * np.diag(diag_S) * V
        L_tn=sum(diag_S)
    else:
        L=L.T
        U,S,V=np.linalg.svd(L,0)
        thresholded_value=S - alpha / 2
        diag_S=thresholded_value*(thresholded_value > 0)
        L_hat=U * np.diag(diag_S) * V
        L_hat=L_hat.T
        L_tn=sum(diag_S)
    return L_hat,L_tn
