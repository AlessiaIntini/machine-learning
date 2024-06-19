import numpy as np
import scipy.optimize


def vcol(x):
    return x.reshape((x.size, 1))


def vrow(x):
    return x.reshape((1, x.size))


class SVM:
    def __init__(self, hparams, kernel=None, prior=0):
        self.kernelType = kernel
        self.C = hparams['C']
        self.K = hparams['K']
        self.eps = hparams.get('eps')
        self.gamma = hparams.get('gamma')
        self.c = hparams.get('c')
        self.d = hparams.get('d')
        self.prior = prior

    def __LDc_obj(self, alpha):
        ones_matrix = np.ones((alpha.shape[0], 1))
        t = 0.5 * np.dot(np.dot(alpha.T, self.H), alpha) - np.dot(alpha.T, ones_matrix).sum(), (
                np.dot(self.H, alpha) - 1).flatten()
        return t

    def __polynomial_kernel(self, X1, X2):
        ker = (np.dot(X1.T, X2) + self.c) ** self.d + self.K ** 2
        return ker

    def __RBF_kernel(self, X1, X2):
        # x = np.repeat(X1, X2.shape[1], axis=1)
        # y = np.tile(X2, X1.shape[1])
        # ker = np.exp(
        #     -self.gamma * np.linalg.norm(x - y, axis=0).reshape(X1.shape[1], X2.shape[1]) ** 2) + self.K ** 2
        # return ker
        D1Norms = (X1 ** 2).sum(0)
        D2Norms = (X2 ** 2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(X1.T, X2)
        return np.exp(-self.gamma * Z)

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.N = Dtrain.shape[1]
        self.Ltrain_z = self.Ltrain * 2 - 1
        self.Ltrain_z_matrix = self.Ltrain_z.reshape(-1, 1) * self.Ltrain_z.reshape(1, -1)
        self.bounds = [(0, self.C) for i in self.Ltrain]

        if self.prior != 0:
            empP = (self.Ltrain == 1).sum() / len(self.Ltrain)
            self.bounds[self.Ltrain == 1] = (0, self.C * self.prior / empP)
            self.bounds[self.Ltrain == 0] = (0, self.C * (1 - self.prior) / (1 - empP))

        if self.kernelType is not None:
            if self.kernelType == 'Polynomial':
                ker = self.__polynomial_kernel(self.Dtrain, self.Dtrain)
            elif self.kernelType == 'RBF':
                ker = self.__RBF_kernel(self.Dtrain, self.Dtrain)
            else:
                return
            self.H = self.Ltrain_z_matrix * ker
        else:
            # self.expandedD = np.vstack((Dtrain, self.K * np.ones(self.N)))
            self.expandedD = np.vstack([Dtrain, np.ones((1, Dtrain.shape[1])) * self.K])
            # G = np.dot(self.expandedD.T, self.expandedD)
            # self.H = G * self.Ltrain_z_matrix
            self.H = np.dot(self.expandedD.T, self.expandedD) * self.Ltrain_z.reshape(self.Ltrain_z.size,
                                                                                      1) * self.Ltrain_z.reshape(1,
                                                                                                                 self.Ltrain_z.size)

        self.alpha, self.primal, _ = scipy.optimize.fmin_l_bfgs_b(func=self.__LDc_obj,
                                                                  bounds=self.bounds,
                                                                  x0=np.zeros(Dtrain.shape[1]),
                                                                  factr=1.0)
        if self.kernelType is None:
            self.wc = np.sum(
                self.alpha.reshape(1, self.alpha.size) * self.Ltrain_z.reshape(1, self.alpha.size) * self.expandedD,
                axis=1)

        self.dual_value = - self.primal
        return self

    def compute_primal_dual_value(self):
        primal_value = 0.5 * np.linalg.norm(self.wc) ** 2 + self.C * np.sum(
            np.maximum(0, 1 - self.Ltrain_z * (np.dot(self.wc.T, self.expandedD))))
        self.primal_value = primal_value
        return self.primal_value, self.dual_value

    def compute_duality_gap(self):
        return self.primal_value - self.dual_value

    def predict(self, Dtest, labels=False):
        if self.kernelType is not None:
            if self.kernelType == 'Polynomial':
                self.S = np.sum(
                    np.dot((self.alpha * self.Ltrain_z).reshape(1, -1), self.__polynomial_kernel(self.Dtrain, Dtest)),
                    axis=0)
            elif self.kernelType == 'RBF':
                self.S = np.sum(
                    np.dot((self.alpha * self.Ltrain_z).reshape(1, -1), self.__RBF_kernel(self.Dtrain, Dtest)), axis=0)
            else:
                return
        else:
            # self.wc = np.sum(self.alpha * self.Ltrain_z * self.expandedD, axis=1)
            # self.w, self.b = self.wc[:-1], self.wc[-1::]
            self.w, self.b = self.wc[0:self.Dtrain.shape[0]], self.wc[-1] * self.K
            # self.S = np.dot(self.w.T, Dtest) + self.b * self.K
            self.S = (vrow(self.w) @ Dtest + self.b).ravel()  # * self.K

        if labels is True:
            predicted_labels = np.where(self.S > 0, 1, 0)
            return predicted_labels
        else:
            return self.S
