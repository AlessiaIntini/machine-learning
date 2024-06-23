import matplotlib.pyplot as plt

import ReadData as rd
import plot


def plot_features(D, L):
    label_name = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']

    mu = rd.vcol(D.mean(axis=1))
    print(mu)
    DC = D - mu
    for i in [0, 2, 4]:
        # hist diagram
        plt.figure(label_name[i] + " and " + label_name[i + 1], figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plot.hist(DC, L, features=i, label=label_name[i])
        plt.subplot(2, 2, 2)
        plot.hist(DC, L, features=i + 1, label=label_name[i + 1])

        # scatter diagram
        # plt.figure(label_name[i]+' vs '+label_name[i+1])
        plt.subplot(2, 2, 3)
        plot.scatter(DC, L, features1=i, features2=i + 1, label1=label_name[i], label2=label_name[i + 1])

        # plt.figure(label_name[i+1]+' vs '+label_name[i])
        plt.subplot(2, 2, 4)
        plot.scatter(DC, L, features1=i + 1, features2=i, label1=label_name[i + 1], label2=label_name[i])
        plt.show()
    # covariance
    C = (D @ D.T) / float(D.shape[1])
    # quadrato della deviazione standard
    var = rd.vcol(D.var(axis=1))  # questa è la diagonale della matrice di covarianza, cio di C
    print('Var', var)
    std = rd.vcol(D.std(axis=1))  # questo è il quadrato della varianza
    print(std)


def plot_single_feature(D, L):
    label_name = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']
    num_features = D.shape[0]

    plt.figure(figsize=(10, 10))

    for i in range(num_features):
        plt.subplot(2, 3, i + 1)

        feature = D[i, :]

        feature_class_0 = feature[L == 0]
        feature_class_1 = feature[L == 1]

        plt.hist(feature_class_0, alpha=0.5, label='Fake')
        plt.hist(feature_class_1, alpha=0.5, label='Genuine')

        plt.title(label_name[i])
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_features_pairwise(D, L):
    label_name = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']
    num_features = D.shape[0]

    plt.figure(figsize=(10, 10))

    for i in range(0, num_features, 2):  # Incrementa l'indice di 2 ogni volta
        plt.subplot(num_features // 2, 1, i // 2 + 1)  # Crea un subplot per ogni coppia di feature

        feature1 = D[i, :]
        feature2 = D[i + 1, :]

        feature1_class_0 = feature1[L == 0]
        feature1_class_1 = feature1[L == 1]
        feature2_class_0 = feature2[L == 0]
        feature2_class_1 = feature2[L == 1]

        plt.scatter(feature1_class_0, feature2_class_0, alpha=0.5, label='Fake')
        plt.scatter(feature1_class_1, feature2_class_1, alpha=0.5, label='Genuine')

        plt.title(f'{label_name[i]} vs {label_name[i + 1]}')
        plt.xlabel(label_name[i])
        plt.ylabel(label_name[i + 1])
        plt.legend()

    plt.tight_layout()  # Ajusta la disposizione dei subplot
    plt.show()
