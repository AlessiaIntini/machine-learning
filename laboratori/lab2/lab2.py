import numpy as np
import matplotlib.pyplot as plt

def load(name):
    f=open(name,'r')
    l_array=np.array([],dtype=int)
    d_array=None
    flowerType=-1
    for line in f:
        val=line.split(',')
        colData=np.array(val[0:4],dtype=float).reshape(4,1)
        if val[4]=='Iris-setosa\n':
            flowerType=0
        elif val[4]=='Iris-versicolor\n':
            flowerType=1
        elif val[4]=='Iris-virginica\n':
            flowerType=2
        l_array=np.append(l_array,flowerType)
        if d_array is None:
            d_array=colData
        else:
            d_array=np.append(d_array,colData,axis=1)
    return d_array,l_array
            
def hist(d_array,l_array,caratteristics,label):
    M0=(l_array==0)
    D0=d_array[caratteristics,M0]
    plt.hist(D0,density=True,label='Iris-setosa')
    M1=(l_array==1)
    D1=d_array[caratteristics,M1]
    plt.hist(D1,density=True,label='Iris-versicolor')
    M2=(l_array==2)
    D2=d_array[caratteristics,M2]
    plt.hist(D2,density=True,label='Iris-virginica')
    plt.xlabel(label)
    plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    # plt.show()
       
def scatter(d_array,l_array,caratteristics1,caratteristics2,label1,label2):
    M0=(l_array==0)
    plt.scatter(d_array[caratteristics1,M0],d_array[caratteristics2,M0],label='Iris-setosa')
    M1=(l_array==1)
    plt.scatter(d_array[caratteristics1,M1],d_array[caratteristics2,M1],label='Iris-versicolor')
    M2=(l_array==2)
    plt.scatter(d_array[caratteristics1,M2],d_array[caratteristics2,M2],label='Iris-virginica')
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    # plt.show()
    
def mcol(array,shape):
    return array.reshape(shape,1)

def mrow(array,shape): #shape è d_array.shape[1]
    return array.reshape(1,shape)

if __name__=='__main__':
    d_array,l_array=load('iris.csv')
    label_name=['sepal length','sepal width','petal length','petal width']
    # for i in range(4):
    #     hist(d_array,l_array,i,label_name[i])
    #     #plt.show()
    #     plt.figure(label_name[i])
    
    # for i in range(0,4):
    #     for j in range(0,4):
    #         if i==j:
    #             continue
    #         scatter(d_array,l_array,i,j,label_name[i],label_name[j])   
    #        # plt.show()
    #         plt.figure(label_name[i]+' vs '+label_name[j])

    #statistics
    #mean
    mu=mcol(d_array.mean(axis=1),d_array.shape[0])
    print(mu)
    DC=d_array-mu
    muDC=mcol(DC.mean(axis=1),DC.shape[0])
    print(muDC)
    
    for i in range(4):
        hist(DC,l_array,i,label_name[i])
        plt.show()
        plt.figure(label_name[i])
    
    for i in range(0,4):
        for j in range(0,4):
            if i==j:
                continue
            scatter(DC,l_array,i,j,label_name[i],label_name[j])   
            plt.show()
            plt.figure(label_name[i]+' vs '+label_name[j])
    #covariance
    C=(DC@DC.T)/float(d_array.shape[1])
    print(C)
    var=mcol(d_array.var(axis=1),shape=d_array.shape[0]) #questa è la diagonale della matrice di covarianza, cio di C
    print(var)
    std=mcol(d_array.std(axis=1),shape=d_array.shape[0]) #questo è il quadrato della varianza
    print(std)

    