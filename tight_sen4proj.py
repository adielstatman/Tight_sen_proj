# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:53:09 2019

@author: Adiel
"""
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix as SM
from scipy.sparse import coo_matrix as CM
import scipy 
from scipy.sparse import dia_matrix
from scipy.sparse import hstack,vstack
import matplotlib.pyplot as plt
import pandas as pd
def Nonuniform(AA0,k,is_pca,eps,spar): 
        """
        non uniform sampling opponent to our algorithm, from
        Varadarajan, Kasturi, and Xin Xiao. "On the sensitivity of shape fitting problems." arXiv preprint arXiv:1209.4893 (2012).‏
        input:
            AA0:data matrix
            k: dimension of the approximated subspace
            is_pca: if 1 will provide a coreset to PCA, 0 will provide coreset for SVD
            eps: detemines coreset size
            spar: is data in sparse format
        output:
            weighted coreset
        """
        d=AA0.shape[1]
        if is_pca==1:
                k=k+1
                AA0=PCA_to_SVD(AA0,eps,spar)
        if is_jl==1:
            dex=int(k*np.log(AA0.shape[0]))
            ran=np.random.randn(AA0.shape[1],dex)
            if spar==1:
                AA=SM.dot(AA0,ran)
            else:
                AA=np.dot(AA0,ran)
        else:
            AA=AA0
        size_of_coreset=int(k+k/eps-1) 
        U,D,VT=ssp.linalg.svds(AA,k)       
        V = np.transpose(VT)
        AAV = np.dot(AA, V)
        del V
        del VT    
        x = np.sum(np.power(AA, 2), 1)
        y = np.sum(np.power(AAV, 2), 1)
        P = np.abs(x - y)
        AAV=np.concatenate((AAV,np.zeros((AAV.shape[0],1))),1)
        Ua, _, _ = ssp.linalg.svds(AAV,k)
        U = np.sum(np.power(Ua, 2), 1)
        pro = 2 * P / np.sum(P) + 8 * U
        if is_pca==1:
            pro=pro+81*eps
        pro0 = pro / sum(pro)
        w=np.ones(AA.shape[0])
        u=np.divide(w,pro0)/size_of_coreset
        DMM_ind=np.random.choice(AA.shape[0],size_of_coreset, p=pro0)
        u1=np.reshape(u[DMM_ind],(len(DMM_ind),1))
        if spar==1:
            SA0=SM(AA0)[DMM_ind,:d].multiply(np.sqrt(u1))
        else:
            SA0=np.multiply(np.sqrt(u1),AA0[DMM_ind,:d])
        return SA0   
def sorted_eig(A):
	eig_vals, eig_vecs =scipy.linalg.eigh(A)  	
	eig_vals_sorted = np.sort(eig_vals)[::-1]
	eig_vecs = eig_vecs.T
	eig_vecs_sorted = eig_vecs[eig_vals.argsort()][::-1]
	return eig_vals_sorted,eig_vecs_sorted
def get_unitary_matrix(n, m):
	a = np.random.random(size=(n, m))
	q, _ = np.linalg.qr(a)
	return q
	
def get_gamma(A_tag,l,d):
	vals , _ = sorted_eig(A_tag)
	sum_up = 0;sum_down = 0
	for i in range (l) : 
		sum_up += vals[d-i -1]
		sum_down += vals[i]
	return (sum_up/sum_down)

def calc_sens(A,p,j,eps):
	
    d=A.shape[1]; l = d-j;
    A_tag = np.dot(A.T , A) ; 
    p = np.reshape(p, (p.shape[0], 1)).T ; 
    p_tag = np.dot(p.T,p) ;
    s_old = -float("inf")
    x = get_unitary_matrix(d, l)
    step = 0  ; stop = False
    gama = get_gamma(A_tag,l,d);
    stop_rule = (gama*eps)/(1-gama)
    s_l = []
    s_old = 0 
    while  step <20000:	
        s_new =  np.trace( np.dot (np.dot(x.T,p_tag) ,x))  / np.trace( np.dot(np.dot(x.T,A_tag) , x  ))
        	
        s_l.append(s_new)
        G = p_tag - s_new*A_tag
        _ , ev = sorted_eig(G)
        x = ev[:l].T
        if s_new - stop_rule < s_old :                
            return max(s_l)
        s_old = s_new 
        step+=1
    return max(s_l)	
def PCA_to_SVD(P,epsi,is_spar):
    """
    equivalent to algorithm 2 in the paper
    input:
        P: data matrix
        epsi: determine coreset size
        is_spar:is data in sparse format
    output:
        weighted coreset
    """
    if is_spar==0:
        r=1+2*np.max(np.sum(np.power(P,2),1))/epsi**4
        P=np.concatenate((P,r*np.ones((P.shape[0],1))),1)
    else:
        P1=SM.copy(P)
        P1.data=P1.data**2
        r=1+2*np.max(np.sum(P1,1))/epsi**4
        P=hstack((P,r*np.ones((P.shape[0],1))))
    return P
def alaa_coreset(wiki0,j,eps,w,is_pca,spar): 
    """
    our algorithm, equivalent to Algorithm 1 in the paper.
    input:
        wiki0:data matrix
        j: dimension of the approximated subspace
        eps: determine coreset size
        w: initial weights
        is_pca: 1 coreset for pca, 0 coreset dor SVD
        spar: is data in sparse format
    output:
        weighted coreset
    """
    coreset_size=j/eps
    dex=int(j*np.log(wiki0.shape[0]))
    d=wiki0.shape[1]
    if is_pca==1:
        j=j+1
        wiki0=PCA_to_SVD(wiki0,eps,spar)
    if is_jl==1:
        ran=np.random.randn(wiki0.shape[1],dex)
        if spar==1:
            wiki=SM.dot(wiki0,ran)	
        else:
            wiki=np.dot(wiki0,ran)	
    else:
        wiki=wiki0
    w=w/wiki.shape[0]
    sensetivities=[]
    jd=j
    w1=np.reshape(w,(len(w),1))
    wiki1=np.multiply(np.sqrt(w1),wiki)
    k=0
    for i,p in enumerate(wiki1) :
        k=k+1
        sensetivities.append(calc_sens(wiki1,p,jd,eps))
    
    p0=np.asarray(sensetivities)
    if is_pca==1:
        p0=p0+81*eps
    indec=np.random.choice(np.arange(wiki.shape[0]),int(coreset_size),p=p0/np.sum(p0)) #sampling according to the sensitivity
    p=p0/np.sum(p0) #normalizing sensitivies
    w=np.ones(wiki.shape[0])
    u=np.divide(np.sqrt(w),p)/coreset_size #caculating new weights
    u1=u[indec]#picking weights of sampled
    u1=np.reshape(u1,(len(u1),1))
    squ=np.sqrt(u1)   
    if spar==1:        
        C=SM(wiki0)[indec,:d].multiply(squ) #weighted coreset
    else:
        C=np.multiply(squ,wiki0[indec,:d])
    return C

def unif_sam(A,j,eps,is_sparse=0):
    """
    uniform sampling
    input:
        A-data matrix
        j: dimension of the approximated subspace
        is_pca: if 1 will provide a coreset to PCA, 0 will provide coreset for SVD
        eps: detemines coreset size
        is sparse: is data in sparse format
    output:
        random subset
    """
    m=j+int(j/eps)-1
    S=A[np.random.choice(A.shape[0],size=m),:]
    return S

def SVD_streaming(Data,j,is_jl,alg,h,spar):
    """
    streaming tree
        Data=data matrix
        j: dimension of the approximated subspace
        is_jl:whether to produce jl transform
        alg:0 unif sampling,1 opponent,2 our
        h: number of floor of the tree
        spar: is data in sparse format
    """
    coreset_size=Data.shape[0]//(2**(h+1))
    gamma=j/(coreset_size-j+1)
    k=0
    T_h= [0] * (h+1)
    for jj in range(np.power(2,h)): #over all of the leaves
        Q=Data[k:k+2*coreset_size,:]
        k=k+2*coreset_size
        if alg==0:
            T=unif_sam(Q,j,gamma) #making a coreset of the leaf
        if alg==1:
            T=alaa_coreset(Q,j,gamma,coreset_size,np.ones(Q.shape[0])/Q.shape[0],is_pca,spar)
        if alg==2:
            T=Nonuniform(Q,j,is_pca,gamma,spar)    
        i=0
        while (i<h)*(type(T_h[i])!=int): #every time the leaf has a neighbor leaf it should merged and reduced
           if spar==0:
               totT=np.concatenate((T,np.asarray(T_h[i])),0)
           else:
               totT=vstack((T,T_h[i]))
           if alg==0:
                T=unif_sam(totT,j,gamma)
           if alg==1:
                T=alaa_coreset(totT,j,gamma,coreset_size,np.ones(totT.shape[0])/totT.shape[0],is_pca,spar)       
           if alg==2:
               T=Nonuniform(totT,j,is_pca,gamma,spar)
           T_h[i]=0
           i=i+1
        T_h[i]=T
        Q=[]        
    if type(T_h[h])==int: #should be remained only the upper one. if not:
        all_levels=[]
        for g in range (h+1): #collecting all leaves which remained on tree.
            if type(T_h[g])!=int:
                if all_levels==[]:
                   all_levels=np.asarray(T_h[g])
                else:
                    all_levels=np.concatenate((all_levels,np.asarray(T_h[g])),0)
    else:
        all_levels=T_h[h] 
    return all_levels

n=1000
d=10
X=np.random.randn(n,d)
coreset_size=100
eps=j/coreset_size
num_of_floors=int(np.log2(n/coreset_size))
w=np.random.rand(n)
is_pca=0
spar=0
j=d//2
is_jl=spar
Y=alaa_coreset(X,j,eps,w,is_pca,spar)   
Z=SVD_streaming(X,j,is_jl,2,num_of_floors,spar)
_,_,VX=np.linalg.svd(X)
_,_,VY=np.linalg.svd(Y)
_,_,VZ=np.linalg.svd(Z)
VXj=VX[:,:j]
VYj=VY[:,:j]
VZj=VZ[:,:j]
VVXT=np.dot(VXj,VXj.T)
VVYT=np.dot(VYj,VYj.T)
VVZT=np.dot(VZj,VZj.T)
XVVXT=np.dot(X,VVXT)
XVVYT=np.dot(X,VVYT)
XVVZT=np.dot(X,VVZT)
errorY=np.abs(np.linalg.norm(X-XVVYT,'fro')/np.linalg.norm(X-XVVXT,'fro')-1)
errorZ=np.abs(np.linalg.norm(X-XVVZT,'fro')/np.linalg.norm(X-XVVXT,'fro')-1)
print('direct error',errorY)
print('tree error',errorZ)


