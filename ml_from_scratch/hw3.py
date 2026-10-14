###### Your ID ######
# Identifier removed for portfolio publication
#####################

import numpy as np


def add_bias_term(X):
    """
    Add a bias term to each sample of the input data.
    """
    bias = np.ones((X.shape[0], 1))
    return np.concatenate((bias, X), axis=1)


class LogisticRegressionGD():
    def __init__(self, learning_rate=0.0001, max_iter=10000, eps=0.000001, random_state=1):
        self.random_state = random_state
        self.w_ = np.nan
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.class_names = None
        self.loss_history_ = None

    def predict_proba(self, X):
        assert X.shape[1] == self.w_.shape[0]

        z = X @ self.w_
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)

        return np.where(
            probs >= threshold,
            self.class_names[1],
            self.class_names[0]
        )

    def BCE_loss(self, X, y):
        y_01 = np.where(y == self.class_names[0], 0, 1)

        probs = self.predict_proba(X)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)

        loss = -np.mean(
            y_01 * np.log(probs) +
            (1 - y_01) * np.log(1 - probs)
        )

        return loss

    def fit(self, X, y):
        self.loss_history_ = []

        self.class_names = np.unique(y)
        y_01 = np.where(y == self.class_names[0], 0, 1)

        np.random.seed(self.random_state)
        self.w_ = 1e-6 * np.random.randn(X.shape[1])

        prev_loss = np.inf

        for _ in range(int(self.max_iter)):

            probs = self.predict_proba(X)

            gradient = (X.T @ (probs - y_01)) / X.shape[0]

            self.w_ -= self.learning_rate * gradient

            curr_loss = self.BCE_loss(X, y)

            self.loss_history_.append(curr_loss)

            if abs(prev_loss - curr_loss) < self.eps:
                break

            prev_loss = curr_loss

        return self


class LogisticRegressionGD_MB(LogisticRegressionGD):
    """
    Logistic Regression Classifier with mini-batch gradient descent.
    """

    def __init__(self, learning_rate=0.0001, eps=0.000001,
                 random_state=1, batch_size=32, num_epochs=100):

        super().__init__(
            learning_rate,
            max_iter=np.inf,
            eps=eps,
            random_state=random_state
        )

        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y, num_epochs=None, batch_size=None):

        if num_epochs is None:
            num_epochs = self.num_epochs

        if batch_size is None:
            batch_size = self.batch_size

        self.class_names = np.unique(y)

        y_01 = np.where(y == self.class_names[0], 0, 1)

        np.random.seed(self.random_state)

        self.w_ = 1e-6 * np.random.randn(X.shape[1])

        self.loss_history_ = []

        n_samples = X.shape[0]

        prev_loss = np.inf

        for _ in range(num_epochs):

            permutation = np.random.permutation(n_samples)

            X_shuffled = X[permutation]
            y_01_shuffled = y_01[permutation]

            for start in range(0, n_samples, batch_size):

                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch_01 = y_01_shuffled[start:end]

                probs = self.predict_proba(X_batch)

                gradient = (
                    X_batch.T @ (probs - y_batch_01)
                ) / X_batch.shape[0]

                self.w_ -= self.learning_rate * gradient

            curr_loss = self.BCE_loss(X, y)

            self.loss_history_.append(curr_loss)

            if abs(prev_loss - curr_loss) < self.eps:
                break

            prev_loss = curr_loss

        return self


def calc_metrics(y_true, y_pred, positive_class):

    metric_dict = dict(
        tp=None,
        fp=None,
        tn=None,
        fn=None,
        tpr=None,
        fpr=None,
        tnr=None,
        fnr=None,
        accuracy=None,
        precision=None,
        risk=None,
        f1=None
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_pos = y_true == positive_class
    true_neg = y_true != positive_class

    pred_pos = y_pred == positive_class
    pred_neg = y_pred != positive_class

    tp = np.sum(true_pos & pred_pos)
    fp = np.sum(true_neg & pred_pos)
    tn = np.sum(true_neg & pred_neg)
    fn = np.sum(true_pos & pred_neg)

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    metric_dict["tp"] = int(tp)
    metric_dict["fp"] = int(fp)
    metric_dict["tn"] = int(tn)
    metric_dict["fn"] = int(fn)

    metric_dict["tpr"] = safe_div(tp, tp + fn)
    metric_dict["fpr"] = safe_div(fp, fp + tn)
    metric_dict["tnr"] = safe_div(tn, tn + fp)
    metric_dict["fnr"] = safe_div(fn, tp + fn)

    metric_dict["accuracy"] = safe_div(
        tp + tn,
        tp + tn + fp + fn
    )

    metric_dict["precision"] = safe_div(tp, tp + fp)

    metric_dict["risk"] = safe_div(
        fp + fn,
        tp + tn + fp + fn
    )

    metric_dict["f1"] = safe_div(
        2 * metric_dict["precision"] * metric_dict["tpr"],
        metric_dict["precision"] + metric_dict["tpr"]
    )

    return metric_dict


def fpr_tpr_per_threshold(y_true, positive_class_probs, positive_class):

    fpr = []
    tpr = []

    prob_thresholds = np.arange(0, 1, 0.01)

    y_true = np.asarray(y_true)
    positive_class_probs = np.asarray(positive_class_probs)

    if not np.any(y_true == positive_class):
        raise ValueError(
            "No entry in y_true equals positive_class."
        )

    if np.all(y_true == positive_class):
        raise ValueError(
            "All labels equal positive_class; ROC needs both positive and negative samples."
        )

    negative_class = y_true[y_true != positive_class][0]

    for threshold in prob_thresholds:

        y_pred = np.where(
            positive_class_probs >= threshold,
            positive_class,
            negative_class
        )

        metrics = calc_metrics(
            y_true,
            y_pred,
            positive_class
        )

        fpr.append(metrics["fpr"])
        tpr.append(metrics["tpr"])

    return fpr, tpr


class OneVsAllClassifier:

    def __init__(self,
                 binary_classifier_class=LogisticRegressionGD_MB,
                 **binary_classifier_kwargs):

        self.binary_classifier_class = binary_classifier_class
        self.binary_classifier_kwargs = binary_classifier_kwargs
        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        self.classifiers_ = []

        for class_label in self.classes_:

            y_binary = np.where(y == class_label, 1, 0)

            classifier = self.binary_classifier_class(
                **self.binary_classifier_kwargs
            )

            classifier.fit(X, y_binary)

            classifier.class_names = np.array([0, 1])

            self.classifiers_.append(classifier)

        return self

    def predict_proba(self, X):

        scores = np.nan * np.ones(
            (X.shape[0], len(self.classes_))
        )

        for j, classifier in enumerate(self.classifiers_):

            scores[:, j] = classifier.predict_proba(X)

        return scores

    def predict(self, X):

        scores = self.predict_proba(X)

        best_class_indices = np.argmax(scores, axis=1)

        return self.classes_[best_class_indices]