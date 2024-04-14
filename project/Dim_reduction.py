import numpy as np
import matplotlib.pyplot as plt
import PCA 
import LDA 
import plot 

def Dim_red(D,L):
    rang=list(range(1,7))
 
    s,P=PCA.PCA_function(D,6)
    s=PCA.PercentageVariance(s)
    #print("Percentuale di varianza:",s)
    plt.figure("Percentuale di varianza")
    plt.plot(rang,s,'o--')
    plt.show()
    DP=np.dot(P.T,D)
   # print("DP",DP)
    plt.figure("PCA and LDA",figsize=(8,8))
    
    plt.subplot(2,2,1)
    plt.title("PCA")
    plot.scatter(DP,L,0,1,"PC1","PC2")
    plt.subplot(2,2,3)
    plot.hist(DP,L,0,"PC1")
    
    W_LDA=LDA.LDA_function(D,L,2)
    WP_LDA=np.dot(W_LDA.T,D)
    plt.subplot(2,2,4)
    plot.hist(WP_LDA,L,0,"1st direction")
    plt.subplot(2,2,2)
    #LDA in first two directions
    plt.title("LDA ")
    plot.scatter(WP_LDA,L,0,1,"1st direction","2nd direction")
    plt.show()
    
def classification(DTR,LTR,DVAL,LVAL):
    W_TR=LDA.LDA_function(DTR,LTR,2)
    DTR_LDA=np.dot(W_TR.T,DTR)
    DVAL_LDA=np.dot(W_TR.T,DVAL)
    
    plt.figure("DTR and DVAL LDA",figsize=(10,8))
    plt.subplot(1,2,1)
    plot.hist(DTR_LDA,LTR,0,"1st direction",bins=10)
    plt.subplot(1,2,2)
    plot.hist(DVAL_LDA,LVAL,0,"1st direction",bins=10)
    plt.show()
    
    # threshold=(DTR_LDA[0,LTR==0]).mean()+(DTR_LDA[0,LTR==1]).mean()/2.0
    best_threshold = None
    best_error_rate = float('inf')

    for threshold in np.linspace(DTR_LDA.min(), DTR_LDA.max(), 100):
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_LDA[0] >= threshold] = 1
        PVAL[DVAL_LDA[0] < threshold] = 0
        diff = np.abs(PVAL - LVAL).sum()
        error_rate = diff / len(LVAL)
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_threshold = threshold
            
    PVAL=np.zeros(shape=LVAL.shape,dtype=np.int32)
    PVAL[DVAL_LDA[0]>=best_threshold]=1
    PVAL[DVAL_LDA[0]<best_threshold]=0
    
    diff=np.abs(PVAL-LVAL).sum()
    print("number of error",diff)
    error_rate = (diff / len(LVAL)) 
    print("Error rate after lda:", error_rate)
    
    
    #apply PCA and after LDA
    s,P=PCA.PCA_function(DTR,1)
    
    DTR_PCA=np.dot(P.T,DTR)
    DVAL_PCA=np.dot(P.T,DVAL)
    
    W_TR=LDA.LDA_function(DTR_PCA,LTR,2)
    
    DTR_LDA=np.dot(W_TR.T,DTR_PCA)
    DVAL_LDA=np.dot(W_TR.T,DVAL_PCA)
    
    plt.figure("DTR and DVAL LDA",figsize=(10,8))
    plt.subplot(1,2,1)
    plot.hist(DTR_LDA,LTR,0,"1st direction",bins=10)
    plt.subplot(1,2,2)
    plot.hist(DVAL_LDA,LVAL,0,"1st direction",bins=10)
    plt.show()
    
    threshold=(DTR_LDA[0,LTR==0].mean()+DTR_LDA[0,LTR==1].mean())/2.0
    
    # best_threshold = None
    # best_error_rate = float('inf')

    # for threshold in np.linspace(DTR_LDA.min(), DTR_LDA.max(), 100):
    #     PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    #     PVAL[DVAL_LDA[0] >= threshold] = 1
    #     PVAL[DVAL_LDA[0] < threshold] = 0
    #     diff = np.abs(PVAL - LVAL).sum()
    #     error_rate = diff / len(LVAL)
    #     if error_rate < best_error_rate:
    #         best_error_rate = error_rate
    #         best_threshold = threshold
            
    PVAL=np.zeros(shape=LVAL.shape,dtype=np.int32)
    PVAL[DVAL_LDA[0]>=threshold]=1
    PVAL[DVAL_LDA[0]<threshold]=0
    
    diff=np.abs(PVAL-LVAL).sum()
    print("number of error",diff)
    print("len(LVAL)",len(LVAL))
    error_rate = (diff / len(LVAL)) 
    print("Error rate after pca and lda:", error_rate)