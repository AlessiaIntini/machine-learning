import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
import plot 
    
def plot_features(D,L):
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

    mu=rd.mcol(D.mean(axis=1),D.shape[0])
    print(mu)
    DC=D-mu
    for i in [0,2,4]:
        #hist diagram
        plt.figure(label_name[i]+" and "+label_name[i+1],figsize=(8,8))
        plt.subplot(2,2,1)
        plot.hist(DC,L,features=i,label=label_name[i])
        plt.subplot(2,2,2)
        plot.hist(DC,L,features=i+1,label=label_name[i+1])
       
    
    #scatter diagram
        #plt.figure(label_name[i]+' vs '+label_name[i+1])
        plt.subplot(2,2,3)
        plot.scatter(DC,L,features1=i,features2=i+1,label1=label_name[i],label2=label_name[i+1])
        
        #plt.figure(label_name[i+1]+' vs '+label_name[i])
        plt.subplot(2,2,4)
        plot.scatter(DC,L,features1=i+1,features2=i,label1=label_name[i+1],label2=label_name[i])
        plt.show()
    #covariance
    C=(D@D.T)/float(D.shape[1]) 
    #quadrato della deviazione standard
    var=rd.mcol(D.var(axis=1),shape=D.shape[0]) #questa è la diagonale della matrice di covarianza, cio di C
    print('Var',var)
    std=rd.mcol(D.std(axis=1),shape=D.shape[0]) #questo è il quadrato della varianza
    print(std)