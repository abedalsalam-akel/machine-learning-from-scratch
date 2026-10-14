import numpy as np
import math
from statistics import NormalDist

def poisson_log_pmf(k, rate):
    log_p = None
    if np.isscalar(k):
        log_p = k * np.log(rate) - rate - math.lgamma(k + 1)
    else:
        k = np.asarray(k)
        log_p = k * np.log(rate) - rate - np.vectorize(math.lgamma)(k + 1)
    return log_p


def poission_analytic_mle(samples):
    mean = None
    mean = np.mean(samples)
    return mean


def poission_confidence_interval(lambda_mle, n, alpha=0.05):
    lower_bound = None
    upper_bound = None
    z = NormalDist().inv_cdf(1 - alpha / 2)
    se = np.sqrt(lambda_mle / n)

    lower_bound = lambda_mle - z * se
    upper_bound = lambda_mle + z * se
    return lower_bound, upper_bound


def get_poisson_log_likelihoods(samples, rates):
    likelihoods = None
    likelihoods = np.array([
        np.sum(poisson_log_pmf(samples, rate))
        for rate in rates
    ])
    return likelihoods


class conditional_independence():

    def __init__(self):

        self.X = {0: 0.3, 1: 0.7}
        self.Y = {0: 0.3, 1: 0.7}
        self.C = {0: 0.5, 1: 0.5}


        self.X_Y_C = {
            (0, 0, 0): 0.005,
            (0, 0, 1): 0.125,
            (0, 1, 0): 0.045,
            (0, 1, 1): 0.125,
            (1, 0, 0): 0.045,
            (1, 0, 1): 0.125,
            (1, 1, 0): 0.405,
            (1, 1, 1): 0.125,
        }

        self.X_Y = {
            (0, 0): 0.13,
            (0, 1): 0.17,
            (1, 0): 0.17,
            (1, 1): 0.53
        }

        self.X_C = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.45,
            (1, 1): 0.25
        }

        self.Y_C = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.45,
            (1, 1): 0.25
        }


    def is_X_Y_dependent(self):
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x in X:
            for y in Y:
                if not np.isclose(X_Y[(x, y)], X[x] * Y[y]):
                    return True
        return False

    def is_X_Y_given_C_independent(self):
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for x in X:
            for y in Y:
                for c in C:
                    lhs = X_Y_C[(x, y, c)] / C[c]

                    px_given_c = X_C[(x, c)] / C[c]
                    py_given_c = Y_C[(y, c)] / C[c]

                    rhs = px_given_c * py_given_c

                    if not np.isclose(lhs, rhs):
                        return False
        return True


def normal_pdf(x, mean, std):
    p = None
    std = np.maximum(std, 1e-12)
    p = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return p


class NaiveNormalClassDistribution():
    def __init__(self, class_value):
        self.class_value = class_value
        self.prior = None
        self.means = None
        self.stds = None

    def fit(self, dataset):
        X = dataset[:, :-1]
        y = dataset[:, -1]

        class_samples = X[y == self.class_value]

        self.prior = class_samples.shape[0] / X.shape[0]
        self.means = np.mean(class_samples, axis=0)
        self.stds = np.std(class_samples, axis=0)

        self.stds = np.where(self.stds == 0, 1e-12, self.stds)
    
    def get_prior(self):
        prior = None
        prior = self.prior
        return prior
    
    def get_instance_likelihood(self, x):
        likelihood = None
        likelihood = np.prod(normal_pdf(x, self.means, self.stds))
        return likelihood
    
    def get_instance_joint_prob(self, x):
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob


class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        pred = None
        p0 = self.ccd0.get_instance_joint_prob(x)
        p1 = self.ccd1.get_instance_joint_prob(x)

        if p0 > p1:
            pred = 0
        else:
            pred = 1
        return pred

    
def multi_normal_pdf(x, mean, cov):
    pdf = None
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    d = len(mean)

    cov = cov + 1e-12 * np.eye(d)

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    diff = x - mean

    coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov)
    exponent = -0.5 * diff.T @ inv_cov @ diff

    pdf = coefficient * np.exp(exponent)
    return pdf


class MultiNormalClassDistribution():

    def __init__(self, class_value):
        self.class_value = class_value
        self.prior = None
        self.mean = None
        self.cov = None

    def fit(self, dataset):
        X = dataset[:, :-1]
        y = dataset[:, -1]

        class_samples = X[y == self.class_value]

        self.prior = class_samples.shape[0] / X.shape[0]
        self.mean = np.mean(class_samples, axis=0)

        centered = class_samples - self.mean
        self.cov = (centered.T @ centered) / class_samples.shape[0]

        self.cov = self.cov + 1e-12 * np.eye(self.cov.shape[0])
        
    def get_prior(self):
        prior = None
        prior = self.prior
        return prior
    
    def get_instance_likelihood(self, x):
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_joint_prob(self, x):
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob


def compute_accuracy(test_set, map_classifier):
    acc = None
    X = test_set[:, :-1]
    y = test_set[:, -1]

    correct = 0

    for i in range(X.shape[0]):
        pred = map_classifier.predict(X[i])
        if pred == y[i]:
            correct += 1

    acc = correct / X.shape[0]
    return acc


class DiscreteNBClassDistribution():
    def __init__(self, class_value):
        self.class_value = class_value
        self.n_class = None
        self.prior = None
        self.feature_values = None
        self.probabilities = None

    def fit(self, dataset):
        X = dataset[:, :-1]
        y = dataset[:, -1]

        class_samples = X[y == self.class_value]

        self.n_class = class_samples.shape[0]
        self.prior = self.n_class / X.shape[0]

        self.feature_values = []
        self.probabilities = []

        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            self.feature_values.append(values)

            probs_j = {}

            for value in values:
                count = np.sum(class_samples[:, j] == value)

                prob = (count + 1) / (self.n_class + len(values))

                probs_j[value] = prob

            self.probabilities.append(probs_j)
    
    def get_prior(self):
        prior = None
        prior = self.prior
        return prior
    
    def get_instance_likelihood(self, x):
        likelihood = None
        likelihood = 1

        for j in range(len(x)):
            value = x[j]

            if value in self.probabilities[j]:
                likelihood *= self.probabilities[j][value]
            else:
                likelihood *= 1 / (self.n_class + len(self.feature_values[j]))
        return likelihood
    
    def get_instance_joint_prob(self, x):
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob