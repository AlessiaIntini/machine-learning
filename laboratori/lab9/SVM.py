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
        # self.eps = hparams.get('eps')
        # self.gamma = hparams.get('gamma')
        # self.c = hparams.get('c')
        # self.d = hparams.get('d')
        self.prior = prior

    def __LDc_obj(self, alpha):
        ones_matrix = np.ones((alpha.shape[0], 1))
        t = 0.5 * np.dot(np.dot(vrow(alpha), self.H), alpha) - np.dot(alpha.T, ones_matrix).sum(), np.dot(self.H,
                                                                                                          alpha) - 1
        return t

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.N = Dtrain.shape[1]
        self.Ltrain_z = self.Ltrain * 2 - 1
        self.Ltrain_z_matrix = self.Ltrain_z.reshape(-1, 1) * self.Ltrain_z.reshape(1, -1)
        self.bounds = np.array([(0, self.C)] * Ltrain.shape[0])

        if self.prior != 0:
            empP = (self.Ltrain == 1).sum() / len(self.Ltrain)
            self.bounds[self.Ltrain == 1] = (0, self.C * self.prior / empP)
            self.bounds[self.Ltrain == 0] = (0, self.C * (1 - self.prior) / (1 - empP))

            # if self.kernelType is not None:
            #      # if self.kernelType == 'Polynomial':
            #     #     ker = self.__polynomial_kernel(self.Dtrain, self.Dtrain)
            #     # elif self.kernelType == 'RBF':
            #     #     ker = self.__RBF_kernel(self.Dtrain, self.Dtrain)
            #     # else:
            #     #     return
            #     # self.H = self.Ltrain_z_matrix * ker
            # else:
            self.expandedD = np.vstack((Dtrain, self.K * np.ones(self.N)))
            G = np.dot(self.expandedD.T, self.expandedD)
            self.H = G * self.Ltrain_z_matrix

        self.alpha, self.primal, _ = scipy.optimize.fmin_l_bfgs_b(func=self.__LDc_obj,
                                                                  bounds=self.bounds,
                                                                  x0=np.zeros(Dtrain.shape[1]),
                                                                  factr=1.0)
        self.wc = np.sum(self.alpha * self.Ltrain_z * self.expandedD, axis=1)
        w = self.wc[:-1]

        self.primalValue = 0.5 * w.T * w + self.C * np.sum(np.maximum(0, 1 - self.Ltrain_z * np.dot(w.T, Dtrain)))

        return self

    def compute_duality_gap(self):
        return self.primalValue - self.primal

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
            self.w, self.b = self.wc[:-1], self.wc[-1::]
            self.S = np.dot(self.w.T, Dtest) + self.b * self.K

        if labels is True:
            predicted_labels = np.where(self.S > 0, 1, 0)
            return predicted_labels
        else:
            return self.S
