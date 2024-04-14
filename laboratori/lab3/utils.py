import numpy as np
import matplotlib.pyplot as plt
def mcol(array,shape):
    return array.reshape(shape,1)

def mrow(array,shape): #shape è d_array.shape[1]
    return array.reshape(1,shape)
def vcol(array):
    return array.reshape(1,array.size)
def vrow(array):
    return array.reshape(array.size,1)

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

def PercentageVariance(EigenValues):
    eigenvalues=EigenValues[::-1]
    #print("Eigenvalues:",eigenvalues)
    sum=np.sum(eigenvalues)
    for i in range(0,len(eigenvalues)):
        M=np.sum(eigenvalues[:i+1])
        ratio=M/sum*100
        #print("Percentuale di varianza spiegata da PC",i+1,":",ratio)
    return M

def hist(d_array,l_array,caratteristics,label,bins=10):
    M0=(l_array==0)
    D0=d_array[caratteristics,M0]
    plt.hist(D0,density=True,label='Iris-setosa',alpha=0.5,bins=bins)
    M1=(l_array==1)
    D1=d_array[caratteristics,M1]
    plt.hist(D1,density=True,label='Iris-versicolor',alpha=0.5,bins=bins)
    M2=(l_array==2)
    D2=d_array[caratteristics,M2]
    plt.hist(D2,density=True,label='Iris-virginica',alpha=0.5,bins=bins)
    plt.xlabel(label)
    plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    # plt.show()
    
def scatter(d_array,l_array,component1, component2,label1,label2):
    M0=(l_array==0)
    plt.scatter(d_array[component1,M0],d_array[component2,M0],label='Iris-setosa',alpha=0.5)
    M1=(l_array==1)
    plt.scatter(d_array[component1,M1],d_array[component2,M1],label='Iris-versicolor',alpha=0.5)
    M2=(l_array==2)
    plt.scatter(d_array[component1,M2],d_array[component2,M2],label='Iris-virginica',alpha=0.5)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
    
    plt.show()
