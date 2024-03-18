import numpy as np
import matplotlib.pyplot as plt

def mcol(array,shape):
    return array.reshape(shape,1)

def mrow(array,shape): #shape è d_array.shape[1]
    return array.reshape(1,shape)

def load(name):
    f=open(name,'r')
    l_array=np.array([],dtype=int)
    d_array=None
    
    for line in f:
        val=line.split(',')
        colData=mcol(np.array(val[0:6],dtype=float),shape=6)
        l_array=np.append(l_array,int(val[6]))
        if d_array is None:
            d_array=colData
        else:
            d_array=np.append(d_array,colData,axis=1)
    return d_array,l_array

def hist(D,L,features,label):
    D0=D[features,L==0]
    print("D0",D0)
    plt.hist(D0,label='False',density=True,alpha=0.5)
    D1=D[features,L==1]
    print("D1",D1)
    plt.hist(D1,label='True',density=True,alpha=0.5)
    plt.xlabel(label)
    plt.legend(['False','True'])
       
def scatter(D,L,features1,features2,label1,label2):
    plt.scatter(D[features1,L==0],D[features2,L==0],label='False',alpha=0.5)
    plt.scatter(D[features1,L==1],D[features2,L==1],label='True',alpha=0.5)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(['False','True'])
    
if __name__=='__main__':
    D,L=load('trainData.txt')
    print("D",D)
    label_name=['feature1','feature2','feature3','feature4','feature5','feature6']
    
    
    # for i in [0,2,4]:
    #     #hist diagram
    #     plt.figure(label_name[i]+label_name[i+1],figsize=(8,8))
    #     plt.subplot(2,2,1)
    #     hist(D,L,features=i,label=label_name[i])
    #     #plt.figure(label_name[i+1])
    #     plt.subplot(2,2,2)
    #     hist(D,L,features=i+1,label=label_name[i+1])
       
    
    # #scatter diagram
    #     #plt.figure(label_name[i]+' vs '+label_name[i+1])
    #     plt.subplot(2,2,3)
    #     scatter(D,L,features1=i,features2=i+1,label1=label_name[i],label2=label_name[i+1])
        
    #     #plt.figure(label_name[i+1]+' vs '+label_name[i])
    #     plt.subplot(2,2,4)
    #     scatter(D,L,features1=i+1,features2=i,label1=label_name[i+1],label2=label_name[i])
    #     plt.show()

    mu=mcol(D.mean(axis=1),D.shape[0])
    print(mu)
    DC=D-mu
    for i in [0,2,4]:
        #hist diagram
        plt.figure(label_name[i]+" and "+label_name[i+1],figsize=(8,8))
        plt.subplot(2,2,1)
        hist(DC,L,features=i,label=label_name[i])
        plt.subplot(2,2,2)
        hist(DC,L,features=i+1,label=label_name[i+1])
       
    
    #scatter diagram
        #plt.figure(label_name[i]+' vs '+label_name[i+1])
        plt.subplot(2,2,3)
        scatter(DC,L,features1=i,features2=i+1,label1=label_name[i],label2=label_name[i+1])
        
        #plt.figure(label_name[i+1]+' vs '+label_name[i])
        plt.subplot(2,2,4)
        scatter(DC,L,features1=i+1,features2=i,label1=label_name[i+1],label2=label_name[i])
        plt.show()
    #covariance
    C=(D@D.T)/float(D.shape[1])
    var=mcol(D.var(axis=1),shape=D.shape[0]) #questa è la diagonale della matrice di covarianza, cio di C
    print('Var',var)
    std=mcol(D.std(axis=1),shape=D.shape[0]) #questo è il quadrato della varianza
    print(std)