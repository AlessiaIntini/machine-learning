import numpy as np
import scipy.special
import scipy.optimize
import BayesDecisionModel as bdm
import Evaluation as e


class QuadraticLogisticRegression:
    def __init__(self, lbd, prior_weighted=False, prior=0.5):
        self.lbd = lbd
        self.prior_weighted = prior_weighted
        self.prior = prior

    def __compute_zi(self, ci):
        return 2 * ci - 1

    def __logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        z = 2 * self.Ltrain - 1
        exp = (np.dot(w.T, self.Dtrain_exp) + b)
        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        avg_risk = (np.logaddexp(0, -exp * z)).mean()
        return reg + avg_risk

    def __logreg_obj_prior_weighted(self, v):
        w, b = v[0:-1], v[-1]
        z = 2 * self.Ltrain - 1
        reg = 0.5 * self.lbd * np.linalg.norm(w) ** 2
        exp = (np.dot(w.T, self.Dtrain_exp) + b)
        avg_risk_0 = np.logaddexp(0, -exp[self.Ltrain == 0] * z[self.Ltrain == 0]).mean() * (1 - self.prior)
        avg_risk_1 = np.logaddexp(0, -exp[self.Ltrain == 1] * z[self.Ltrain == 1]).mean() * self.prior
        return reg + avg_risk_0 + avg_risk_1

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.F = Dtrain.shape[0]
        self.K = len(set(Ltrain))
        self.N = Dtrain.shape[1]
        self.Dtrain_exp = self.__expand_features_space(Dtrain)
        obj_function = self.__logreg_obj if self.prior_weighted is False else self.__logreg_obj_prior_weighted
        self.x, f, d = scipy.optimize.fmin_l_bfgs_b(func=obj_function,
                                                    x0=np.zeros(self.Dtrain_exp.shape[0] + 1),
                                                    approx_grad=True,
                                                    # iprint=0
                                                    )
        return self.x

    def __vectorize(self, M):
        M_vec = np.hstack(M).reshape(-1, 1)
        return M_vec

    def __expand_features_space(self, D):
        D_exp = np.zeros(shape=(self.F * self.F + self.F, D.shape[1]))
        for i in range(D.shape[1]):
            xi = D[:, i:i + 1]
            D_exp[:, i:i + 1] = np.vstack((self.__vectorize(np.dot(xi, xi.T)), xi))
        return D_exp

    def predict(self, Dtest, label=True):
        w, b = self.x[0:-1], self.x[-1]
        Dtest_exp = self.__expand_features_space(Dtest)
        S = np.zeros((Dtest_exp.shape[1]))
        for i in range(Dtest_exp.shape[1]):
            xi = Dtest_exp[:, i:i + 1]
            s = np.dot(w.T, xi) + b
            S[i] = s
        if label:
            LP = S > 0
            return LP
        else:
            return S

    def predictThreshold(self, Dtest, threshold):
        w = self.x[:-1]
        b = self.x[-1]
        sval = np.dot(w.T, self.__expand_features_space(Dtest)) + b

        return np.int32(sval > threshold)

    def calculateS(self, DVAL):
        w = self.x[:-1]
        b = self.x[-1]
        sval = np.dot(w.T, self.__expand_features_space(DVAL)) + b
        return sval

    def compute_minDCF_actDCF(self, LVAL, DVAL, pi_emp, Cfn=1, Cfp=1, prior=0.5):
        w = self.x[:-1]
        b = self.x[-1]
        sval = np.dot(w.T, self.__expand_features_space(DVAL)) + b
        predictedLabels = np.int32(sval > 0)
        error_rate = e.error_rate(predictedLabels, LVAL)
        print("Error rate:", error_rate, "%")
        sValLLR = sval - np.log(pi_emp / (1 - pi_emp))
        th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
        predictedLabels = np.int32(sval > th)
        minDCF = bdm.compute_minDCF_binary(sValLLR, LVAL, prior, Cfn, Cfp)
        confusionMatrix = bdm.compute_confusion_matrix(predictedLabels, LVAL)
        actDCF = bdm.computeDCF_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
        print("minDCF:", minDCF)
        print("actDCF:", actDCF)
        return minDCF, actDCF
