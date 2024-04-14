import numpy as np
import scipy.linalg
import utils as ut
import matplotlib.pyplot as plt
import scipy
import sklearn.datasets 
def PCA(D,m):
    
    mu=0
    C=0
    mu=D.mean(axis=1)#è un vettore, cioè una matrice riga
    DC=D-ut.mcol(mu,mu.size)#per centrare i dati
    C=np.dot(DC,DC.T)/float(D.shape[0])#matrice di covarianza
    
    s,U=np.linalg.eigh(C)
    P=U[:,::-1][:,0:m]#matrice di proiezione
    # U,s,Vh=np.linalg.svd(C)
    # P=U[:,0:m]#matrice di proiezione
    return mu,C,P,s

def load_iris():
    return sklearn.datasets.load_iris()['data'].T,sklearn.datasets.load_iris()['target']

def compute_Sv_Sb(D,L,n):
    
    if(n==1):
        num_classes=L.max()+1
       
    else: 
        L = np.where(L == 1, 0, L)
        L = np.where(L == 2, 1, L)
        num_classes=len(set(L))
        
    #separate the data into classes
    
    #number of elements for each class
    D_c=[D[:,L==i] for i in range(num_classes)]
    n_c=[D_c[i].shape[1] for i in range(num_classes)]
    #mean for all the data
    mu=D.mean(1)
    mu=ut.mcol(mu,mu.size)
    
    #mean for each class
    mu_c=[ut.mcol(D_c[i].mean(1),D_c[i].mean(1).size) for i in range(len(D_c))]
    
    S_w,S_b=0,0
    for i in range(num_classes):
        Dc=D_c[i]-mu_c[i]
        #print("Dc",Dc)
        C_i=np.dot(Dc,Dc.T)/Dc.shape[1]
        S_w+=n_c[i]*C_i
        diff=mu_c[i]-mu
        S_b+=n_c[i]*np.dot(diff,diff.T)
        
    S_w/=D.shape[1]
    S_b/=D.shape[1]
    return S_w,S_b
    
def LDA(D,L,m,n):
    #compute Sw and Sb
    Sw,Sb=compute_Sv_Sb(D,L,n)
    #print("Sw",Sw)
    #print("Sb",Sb)
    #compute the eigenvalues and eigenvectors of Sw^-1*Sb
    s,U=scipy.linalg.eigh(Sb,Sw)
    #print("s",s)
    #print("U",U)
    W=U[:,::-1][:,0:m]
    print("W",W)
    # Uw,_,_=np.linalg.svd(W)
    # U=Uw[:,0:m]
    
    # U,s,_=np.linalg.svd(Sw)
    # U=U*ut.vrow(1.0/s**0.5)
    
    
    # P1=np.dot(np.dot(U,np.diag(1.0/s**0.5)),U.T)
    # Sbt=np.dot(np.dot(P1,Sb),P1.T)
    # s,P2=scipy.linalg.eigh(Sbt)
    
    # W=np.dot(P1.T,P2)
    # W= W[:,::-1][:,0:m]

    return np.dot(W.T,D),W

def split_db_2to1(D,L,seed=0):
    nTrain=int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx=np.random.permutation(D.shape[1])
    idxTrain=idx[0:nTrain]
    idxTest=idx[nTrain:]
    
    DTR=D[:,idxTrain]
    DVAL=D[:,idxTest]
    LTR=L[idxTrain]
    LVAL=L[idxTest]
    
    return (DTR,LTR),(DVAL,LVAL)
    

    
if __name__=='__main__':
    PCAD=np.load("IRIS_PCA_matrix_m4.npy")
    LDAD=np.load("IRIS_LDA_matrix_m2.npy")
    rang=list(range(1,4))
    #print("PCAD",PCAD)
    print("LDAD",LDAD)
    D,L=ut.load('iris.csv')
    
    #PCA
    # mu,C,P,s=PCA(D,4)
    # DP=np.dot(P.T,D)
    # s=ut.PercentageVariance(s)
    # # plt.figure("Percentuale di varianza")
    # # plt.plot(rang,s,'o--')
    # # plt.show()  
    
    # plt.figure("PCA",figsize=(10,5))
    # plt.subplot(1,2,1)
    # ut.hist(DP,L,0,"PC1")
    # plt.subplot(1,2,2)
    # ut.scatter(DP,L,0,1,"PC1","PC2")
    
    #LDA
    # WP,W=LDA(D,L,2,1)
    # print("LDA")
    # print("WP",WP)
    # print("W",W)
    # #print("W",W)
    # plt.figure("LDA",figsize=(10,5))
    # plt.subplot(1,2,1)
    # ut.hist(WP,L,0,"1st direction")
    # plt.subplot(1,2,2)
    # #LDA in first two directions
    # ut.scatter(WP,L,0,1,"1st direction","2nd direction")
    
    #classification
    DIris,LIris=load_iris()
    D=DIris[:,LIris!=0]
    L=LIris[LIris!=0]
    print("D",D)
    print("L",L)
    (DTR,LTR),(DVAL,LVAL)=split_db_2to1(D,L)
    # WP_TR,W_TR=LDA(DTR,LTR,2,0)
    # print("LDA classificazione")
    # # print("LTR",LTR)
    # plt.figure("LDA (DTR,LTR)",figsize=(10,5))
    # ut.hist(WP_TR,LTR,0,"1st direction",bins=5)
    # plt.show()
   
    # WP_VAL=np.dot(W_TR.T,DVAL)
    # plt.figure("LDA (DVAL,LVAL)",figsize=(10,5))
    # ut.hist(WP_VAL,LVAL,0,"1st direction",bins=5)
    # plt.show()
    
    # threshold=(WP_TR[0,LTR==1].mean()+WP_TR[0,LTR==2].mean())/2.0
    # PVAL=np.zeros(shape=LVAL.shape,dtype=np.int32)
    # PVAL[WP_VAL[0]>=threshold]=2
    # PVAL[WP_VAL[0]<threshold]=1
    
    # print("PVAL",PVAL)
    # print("LVAL",LVAL)
    # diff=np.abs(PVAL-LVAL).sum()
    # print("number of error",diff)
    
    #classification with PCA
    mu,C,P,s=PCA(DTR,2)
    
    DTR_PCA=np.dot(P.T,DTR)
    DVAL_PCA=np.dot(P.T,DVAL)
    
    DTR_LDA,W_TR=LDA(DTR_PCA,LTR,2,0)
    DVAL_LDA=np.dot(W_TR.T,DVAL_PCA)
    
    plt.figure("LDA with PCA (DTR,LTR)",figsize=(10,5))
    ut.hist(DTR_LDA,LTR,0,"1st direction",bins=5)
    plt.show()
    
    plt.figure("LDA with PCA (DVAL,LVAL)",figsize=(10,5))
    ut.hist(DVAL_LDA,LVAL,0,"1st direction",bins=5)
    plt.show()
    
    threshold=(DTR_LDA[0,LTR==1].mean()+DTR_LDA[0,LTR==2].mean())/2.0
    PVAL=np.zeros(shape=LVAL.shape,dtype=np.int32)
    PVAL[DVAL_LDA[0]>=threshold]=2
    PVAL[DVAL_LDA[0]<threshold]=1
    
    print("PVAL",PVAL)
    print("LVAL",LVAL)
    diff=np.abs(PVAL-LVAL).sum()
    print("number of error",diff)
    
    
    
    