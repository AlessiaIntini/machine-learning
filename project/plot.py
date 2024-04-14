import matplotlib.pyplot as plt

def hist(D,L,features,label,bins=10):
    D0=D[features,L==0]
    plt.hist(D0,label='False',density=True,alpha=0.5,bins=bins)
    D1=D[features,L==1]
    plt.hist(D1,label='True',density=True,alpha=0.5,bins=bins)
    plt.xlabel(label)
    plt.legend(['False','True'])
       
def scatter(D,L,features1,features2,label1,label2):
    plt.scatter(D[features1,L==0],D[features2,L==0],label='False',alpha=0.5)
    plt.scatter(D[features1,L==1],D[features2,L==1],label='True',alpha=0.5)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(['False','True'])