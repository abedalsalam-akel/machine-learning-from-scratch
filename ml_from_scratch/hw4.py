import numpy as np
import copy


def sample_data(n_dimensions, class_means, n_left, n_right, seed=42):
    class_means = np.asarray(class_means, dtype=float)
    if class_means.shape != (2, n_dimensions):
        raise ValueError("class_means must have shape (2, n_dimensions)")

    rng = np.random.default_rng(seed)
    cov = np.eye(n_dimensions)

    X_left = rng.multivariate_normal(class_means[0], cov, size=n_left)
    X_right = rng.multivariate_normal(class_means[1], cov, size=n_right)

    X = np.vstack([X_left, X_right])
    y = np.hstack([-np.ones(n_left, dtype=int), np.ones(n_right, dtype=int)])

    return X, y


class LinearSVM(object):
    def __init__(self, learning_rate=0.05, C=1.0, max_iter=100000, random_state=42, eps=1e-10):
        self.learning_rate = learning_rate
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.loss_history = [np.inf]
        self.eps = eps
        self.w = None
        self.w0 = None

    def predict_raw(self, X):
        return X @ self.w + self.w0

    def predict(self, X):
        raw_predictions = self.predict_raw(X)
        return 2.0 * (raw_predictions >= 0) - 1.0

    def fit(self, X, y, verbose=False):
        np.random.seed(self.random_state)

        n_features = X.shape[1]

        self.w = np.random.random((n_features, 1))
        self.w0 = np.random.random()

        y = np.array(y).reshape(-1, 1)
        y = np.where(y == 0, -1, y)
        self.loss_history = [np.inf]

        for iteration in range(self.max_iter):
            raw_predictions = self.predict_raw(X)
            loss, dw, dw0 = self.compute_hinge_loss_gradients(X, y, raw_predictions)

            if (iteration + 1) % 100 == 0 and verbose:
                print(f"Iteration {iteration + 1}: Loss = {loss:.6f}")

            self.w -= self.learning_rate * dw
            self.w0 -= self.learning_rate * dw0

            if abs(self.loss_history[-1] - loss) < self.eps:
                break

            self.loss_history.append(loss)

    def compute_hinge_loss_gradients(self, X, y, raw_predictions):
        n_examples = X.shape[0]

        margins = 1 - y * raw_predictions
        hinge_losses = np.maximum(0, margins)

        loss = 0.5 * np.sum(self.w ** 2) + (self.C / n_examples) * np.sum(hinge_losses)

        active = (margins > 0).astype(float)

        dw = self.w - (self.C / n_examples) * (X.T @ (active * y))
        dw0 = - (self.C / n_examples) * np.sum(active * y)

        return loss, dw, dw0


def is_separating_hyperplane(X_train, y_train, C):
    svm = LinearSVM(C=C)
    svm.fit(X_train, y_train)

    y_train = np.asarray(y_train).reshape(-1, 1)
    y_train = np.where(y_train == 0, -1, y_train)

    raw_predictions = svm.predict_raw(X_train)

    is_separating = bool(np.all(y_train * raw_predictions > 0))
    return is_separating


def cross_validation(X, y, n_folds, classifier, random_state=42):
    np.random.seed(random_state)

    X = X.copy()
    y = np.reshape(y.copy(), (-1, 1))

    data_Xy = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data_Xy)
    X = data_Xy[:, :-1]
    y = data_Xy[:, -1].reshape(-1, 1)

    X_folds = np.array_split(X, n_folds)
    y_folds = np.array_split(y, n_folds)

    scores = []

    for i in range(n_folds):
        X_val = X_folds[i]
        y_val = y_folds[i]

        X_train = np.vstack([X_folds[j] for j in range(n_folds) if j != i])
        y_train = np.vstack([y_folds[j] for j in range(n_folds) if j != i])

        clf = copy.deepcopy(classifier)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val).reshape(-1, 1)
        acc = np.mean(y_pred == y_val)
        scores.append(acc)

    accuracy = float(np.mean(scores))
    return accuracy


def solve_QP(P, q, G=None, h=None, A=None, b=None, verbose=False):
    try:
        from importlib import import_module
        cp = import_module("cvxpy")
    except ImportError as exc:
        raise ImportError("solve_QP requires cvxpy. Install it with `pip install cvxpy`.") from exc

    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).ravel()
    P = 0.5 * (P + P.T)

    z = cp.Variable(q.shape[0])
    objective = cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(P)) + q @ z)
    constraints = []

    if G is not None and h is not None:
        constraints.append(np.asarray(G, dtype=float) @ z <= np.asarray(h, dtype=float).ravel())
    if A is not None and b is not None:
        constraints.append(np.asarray(A, dtype=float) @ z == np.asarray(b, dtype=float).ravel())

    problem = cp.Problem(objective, constraints)
    for solver in ("OSQP", "CLARABEL", "SCS"):
        try:
            problem.solve(solver=solver, verbose=verbose)
        except cp.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return np.asarray(z.value, dtype=float).ravel()

    raise ValueError(f"QP solver failed with status: {problem.status}")


class KernelSVM(object):
    def __init__(self, C=1.0, kernel='rbf', degree=3, coef0=1.0, gamma=None, tol=1e-5):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol

        self.lambda_star = None
        self.support_vectors_ = None
        self.w0 = None
        self._gamma = None

    def _labels_to_pm1(self, y):
        y = np.asarray(y).ravel()
        return np.where(y == 0, -1.0, y.astype(float))

    def predict_raw(self, X):
        X = np.asarray(X, dtype=float)
        K = self._kernel(X, self.support_vectors_)
        return K @ self.w + self.w0

    def predict(self, X):
        raw_predictions = self.predict_raw(X)
        return 2.0 * (raw_predictions >= 0) - 1.0

    def score(self, X, y):
        y = self._labels_to_pm1(y)
        y_hat = self.predict(X)
        return float(np.mean(y_hat == y))

    def _kernel(self, x, z):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        kernel = self.kernel.lower()

        if kernel in ('poly', 'polynomial'):
            return (self._gamma * (x @ z.T) + self.coef0) ** self.degree

        if kernel == 'rbf':
            X_norm = np.sum(x ** 2, axis=1)[:, None]
            Z_norm = np.sum(z ** 2, axis=1)[None, :]
            distances = X_norm + Z_norm - 2.0 * (x @ z.T)
            return np.exp(-self._gamma * distances)

        raise ValueError("kernel must be one of {'rbf', 'poly', 'polynomial'}")

    def _lambda_to_weights(self, X, y, lambda_star):
        y = self._labels_to_pm1(y)
        lambda_star = np.asarray(lambda_star).ravel()

        support_mask = lambda_star > self.tol
        support_vectors = X[support_mask]

        w = lambda_star[support_mask] * y[support_mask]

        margin_mask = (lambda_star > self.tol) & (lambda_star < self.C - self.tol)

        if not np.any(margin_mask):
            margin_mask = support_mask

        K_margin_support = self._kernel(X[margin_mask], support_vectors)
        w0_values = y[margin_mask] - K_margin_support @ w
        w0 = float(np.mean(w0_values))

        return w, w0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._labels_to_pm1(y)
        n_examples = X.shape[0]
        self._gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]

        K = self._kernel(X, X)

        P = (y[:, None] * y[None, :]) * K
        q = -np.ones(n_examples)

        G = np.vstack([
            -np.eye(n_examples),
             np.eye(n_examples)
        ])

        h = np.hstack([
            np.zeros(n_examples),
            self.C * np.ones(n_examples)
        ])

        A = y.reshape(1, -1)
        b = np.array([0.0])

        lambda_star = solve_QP(P, q, G, h, A, b)

        self.lambda_star = lambda_star
        self.support_mask = lambda_star > self.tol
        self.support_vectors_ = X[self.support_mask]

        self.w, self.w0 = self._lambda_to_weights(X, y, lambda_star)

        return self