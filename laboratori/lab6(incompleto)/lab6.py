import numpy
import scipy

import load


def mcol(v):
    return v.reshape((v.size, 1))


def S1_buildDictionary(lTercets):
    '''
    Create a set of all words contained in the list of tercets lTercets
    lTercets is a list of tercets (list of strings)
    '''

    sDict = set([])
    for s in lTercets:
        words = s.split()
        for w in words:
            sDict.add(w)
    return sDict


def S1_estimateModel(hlTercets, eps=0.1):
    '''
    Build frequency dictionaries for each class.

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: dictionary h_clsLogProb whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    '''

    # Build the set of all words appearing at least once in each class
    sDictCommon = set([])

    # qui ci sono tutte la parole che compaiono in  tutti i terzetti, una sola volta
    for cls in hlTercets:
        lTercets = hlTercets[cls]
        sDictCls = S1_buildDictionary(lTercets)
        sDictCommon = sDictCommon.union(sDictCls)

    # Initializza il dizionario con per ogni parola mette come occorrenza il valore di epsilon lo fai
    # perchè se no poi calcolando il log per le parole che non ci sono verrebbe uguale a inf il risultato
    h_clsLogProb = {}
    for cls in hlTercets:  # Loop over class labels
        h_clsLogProb[cls] = {w: eps for w in
                             sDictCommon}

    # Estimate counts
    for cls in hlTercets:  # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets:  # Loop over all tercets of the class
            words = tercet.split()
            for w in words:  # Loop over words of the given tercet
                # incrementi per ogni occorrenza che trovi
                h_clsLogProb[cls][w] += 1

    # Compute frequencies, and convert them to log-probabilities
    for cls in hlTercets:  # Loop over class labels
        nWordsCls = sum(h_clsLogProb[
                            cls].values())  # somma di tutte le occorrenze delle parole di quel terzetto
        print(nWordsCls)
        for w in h_clsLogProb[cls]:  # Loop over all words
            h_clsLogProb[cls][w] = numpy.log(h_clsLogProb[cls][w]) - numpy.log(nWordsCls)  # Compute log N_{cls,w} / N

    return h_clsLogProb


def compute_classPosteriors(S, logPrior=None):
    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = numpy.log(numpy.ones(S.shape[0]) / float(S.shape[0]))
    J = S + mcol(logPrior)  # Compute joint probability
    ll = scipy.special.logsumexp(J, axis=0)  # Compute marginal likelihood log f(x)
    P = J - ll  # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return numpy.exp(P)


def S1_compute_logLikelihoods(h_clsLogProb, text):
    '''
    Compute the array of log-likelihoods for each class for the given text
    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    The function returns a dictionary of class-conditional log-likelihoods
    '''

    logLikelihoodCls = {cls: 0 for cls in h_clsLogProb}
    for cls in h_clsLogProb:  # Loop over classes
        for word in text.split():  # Loop over words
            if word in h_clsLogProb[cls]:
                logLikelihoodCls[cls] += h_clsLogProb[cls][word]
    return logLikelihoodCls


def S1_compute_logLikelihoodMatrix(h_clsLogProb, lTercets, hCls2Idx=None):
    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used

    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls: idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    S = numpy.zeros((len(h_clsLogProb), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        hScores = S1_compute_logLikelihoods(h_clsLogProb, tercet)
        for cls in h_clsLogProb:  # We sort the class labels so that rows are ordered according to alphabetical order of labels
            clsIdx = hCls2Idx[cls]
            S[clsIdx, tIdx] = hScores[cls]

    return S


def compute_accuracy(P, L):
    '''
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    '''

    PredictedLabel = numpy.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect) / float(NTotal)


if __name__ == '__main__':
    # Load the tercets and split the lists in training and test lists

    lInf, lPur, lPar = load.load_data()

    lInf_train, lInf_evaluation = load.split_data(lInf, 4)
    lPur_train, lPur_evaluation = load.split_data(lPur, 4)
    lPar_train, lPar_evaluation = load.split_data(lPar, 4)

    # print(lInf_train)
    # print(lPur_train)
    # print(lPar_train)

    hClsIdx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    # dici a chi corrispondono le etichette
    hlTercetsTrain = {
        'inferno': lInf_train,
        'purgatorio': lPur_train,
        'paradiso': lPar_train
    }

    # questa è una lista di terzetti
    lTercetsEvaluation = lInf_evaluation + lPur_evaluation + lPar_evaluation

    # questa è la matriche che contiene per ogni classe e ogni parola con che frequenza appare in quella classe
    S1_model = S1_estimateModel(hlTercetsTrain, eps=0.001)

    # questo ti fa ottenere la matrice in cui ogni valore ti dice la probabilità che una parola appartenga a una classe(un terzetto)
    S1_predictions = compute_classPosteriors(
        S1_compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEvaluation,
            hClsIdx,
        ),
        # Pc
        numpy.log(numpy.array([1. / 3., 1. / 3., 1. / 3.]))
    )

    labelsInf = numpy.zeros(len(lInf_evaluation))
    labelsInf[:] = hClsIdx['inferno']

    labelsPar = numpy.zeros(len(lPar_evaluation))
    labelsPar[:] = hClsIdx['paradiso']

    labelsPur = numpy.zeros(len(lPur_evaluation))
    labelsPur[:] = hClsIdx['purgatorio']

    labelsEval = numpy.hstack([labelsInf, labelsPur, labelsPar])

    # Per-class accuracy
    print('Multiclass - S1 - Inferno - Accuracy: %.2f%%' % (
            compute_accuracy(S1_predictions[:, labelsEval == hClsIdx['inferno']],
                             labelsEval[labelsEval == hClsIdx['inferno']]) * 100))
    print('Multiclass - S1 - Purgatorio - Accuracy: %.2f%%' % (
            compute_accuracy(S1_predictions[:, labelsEval == hClsIdx['purgatorio']],
                             labelsEval[labelsEval == hClsIdx['purgatorio']]) * 100))
    print('Multiclass - S1 - Paradiso - Accuracy: %.2f%%' % (
            compute_accuracy(S1_predictions[:, labelsEval == hClsIdx['paradiso']],
                             labelsEval[labelsEval == hClsIdx['paradiso']]) * 100))
