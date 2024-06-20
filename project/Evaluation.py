import numpy as np


def error_rate(PVAL, LVAL):
    return ((PVAL != LVAL).sum() / float(LVAL.size) * 100)


def znormalized_features_training(DT):
    DTmean = DT.mean(axis=1).reshape(-1, 1)
    DTstdDev = DT.std(axis=1).reshape(-1, 1)
    ZnormDT = (DT - DTmean) / DTstdDev
    return ZnormDT


def znormalized_features_evaluation(DE, DT):
    DTmean = DT.mean(axis=1).reshape(-1, 1)
    DTstdDev = DT.std(axis=1).reshape(-1, 1)
    ZnormDE = (DE - DTmean) / DTstdDev
    return ZnormDE


def extract_train_val_folds_from_ary(X, idx, KFOLD=5):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]
