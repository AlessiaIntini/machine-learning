import matplotlib.pyplot as plt
import numpy as np

import GaussianDensity as gd
import ReadData as ut
import BayesDecisionModel as bdm


def hist(D, L, features, label, bins=10):
    D0 = D[features, L == 0]
    plt.hist(D0, label='False', density=True, alpha=0.5, bins=bins)
    D1 = D[features, L == 1]
    plt.hist(D1, label='True', density=True, alpha=0.5, bins=bins)
    plt.xlabel(label)
    plt.legend(['False', 'True'])


def scatter(D, L, features1, features2, label1, label2):
    plt.scatter(D[features1, L == 0], D[features2, L == 0], label='False', alpha=0.5)
    plt.scatter(D[features1, L == 1], D[features2, L == 1], label='True', alpha=0.5)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(['False', 'True'])


def plot_gaussian_density(D, m_ML, C_ML):
    plt.hist(D.ravel(), bins=50, density=True, alpha=0.5)
    XPlot = np.linspace(-8, 12, 1000)

    plt.plot(XPlot.ravel(), np.exp(gd.logpdf_GAU_ND(ut.vrow(XPlot), m_ML, C_ML)))
    plt.xlim(-5, 5)


def plot_minDCF_actDCF(minDCF, actDCF, title, xArray, m=0, xlabel='lambda', One=False):
    if One == False:
        plt.figure()
        if m != 0:
            title = title + " m = " + str(m)
        plt.title(title)
        plt.scatter(xArray, minDCF, label='minDCF', color='b')
        plt.plot(xArray, minDCF, color='b')
        plt.xlabel(xlabel)
        plt.ylabel('minDCF value')
        plt.xscale('log', base=10)
        plt.legend()
        plt.show()

        plt.figure()
        plt.title(title)
        plt.scatter(xArray, actDCF, label='actDCF', color='r')
        plt.plot(xArray, actDCF, color='r')
        plt.xlabel(xlabel)
        plt.ylabel('actDCF value')
        plt.xscale('log', base=10)
        plt.legend()
        plt.show()
    else:
        plt.figure()
        plt.title(title)
        plt.scatter(xArray, minDCF, label='minDCF', color='b')
        plt.plot(xArray, minDCF, color='b')

        plt.scatter(xArray, actDCF, label='actDCF', color='r')
        plt.plot(xArray, actDCF, color='r')
        plt.xlabel(xlabel)
        plt.ylabel('actDCF value')
        plt.xscale('log', base=10)
        plt.legend()
        plt.show()


def bayesPlot(S, L, left=-3, right=3, npts=21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        th = -np.log((effPrior * 1) / ((1 - effPrior) * 1))
        predictedLabels = np.int32(S > th)
        confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, L)
        actDCF_current = bdm.computeDCF_Binary(confusionMatrix, effPrior, 1, 1, normalize=True)
        actDCF.append(actDCF_current)
        minDCF.append(bdm.compute_minDCF_binary(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF
