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
