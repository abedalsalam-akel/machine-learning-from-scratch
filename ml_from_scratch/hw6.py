import numpy as np


def get_random_centroids(X, k, random_state=None):
    '''
    Each centroid is a point in RGB space (color) in the image.
    This function should randomly sample `k` different pixels from the input
    image as the initial centroids for the K-means algorithm.
    The selected `k` pixels should be sampled uniformly from all sets
    of `k` pixels in the image.
    Input: a single image of shape `(num_pixels, 3)`, `k`, the number of centroids,
    and optional `random_state` for reproducible random sampling.
    Notice we are flattening the image to a two dimensional array.
    Output: Randomly chosen centroids of shape `(k, 3)` as a numpy array.
    '''
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X = np.asarray(X)

    rng = np.random.RandomState(random_state)
    selected_indices = rng.choice(X.shape[0], size=k, replace=False)

    centroids = X[selected_indices].copy()

    # make sure you return a numpy array
    return np.asarray(centroids)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def l_p_dist_from_centroids(X, centroids, p=2):
    '''
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids of shape (k, 3)
    The parameter p for the L_p norm distance measure.

    Output: numpy array of shape `(k, num_pixels)`,
    in which entry [j, i] holds the distance of the i-th pixel from the j-th centroid.
    '''
    distances = None
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    differences = np.abs(
        centroids[:, np.newaxis, :] - X[np.newaxis, :, :]
    )

    distances = np.sum(
        differences ** p,
        axis=2
    ) ** (1.0 / p)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances


def kmeans(X, k, p, max_iter=100, epsilon=1e-8, random_state=None):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the L_p distance measure.
    - max_iter: the maximum number of iterations to perform.
    - epsilon: the threshold for convergence.
    - random_state: optional random seed for centroid initialization and empty-cluster handling.

    Outputs:
    - The final centroids as a numpy array of shape `(k, 3)`.
    - The final assignment of all pixels to the closest centroids as a 1-D
      integer numpy array of shape `(num_pixels,)`.
    - The final WCSS as a float.

    If a cluster has no assigned pixels in an update step, randomly choose one
    datapoint from X and set it as the new centroid of that cluster.
    """
    cluster_assignments = None
    centroids = get_random_centroids(X, k, random_state=random_state)
    WCSS = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    rng = np.random.RandomState(random_state)

    # Advance the generator through the same sampling performed during
    # centroid initialization. Empty-cluster handling then continues from
    # the same reproducible random sequence.
    rng.choice(X.shape[0], size=k, replace=False)

    for _ in range(max_iter):
        # Assignment step: assign each pixel to its closest centroid.
        distances = l_p_dist_from_centroids(X, centroids, p=p)
        cluster_assignments = np.argmin(distances, axis=0)

        # Update step: calculate the mean of each cluster.
        new_centroids = np.empty_like(centroids, dtype=float)

        for cluster_idx in range(k):
            cluster_mask = cluster_assignments == cluster_idx

            if np.any(cluster_mask):
                new_centroids[cluster_idx] = np.mean(
                    X[cluster_mask],
                    axis=0
                )
            else:
                random_idx = rng.choice(X.shape[0])
                new_centroids[cluster_idx] = X[random_idx]

        # Compute the L_p movement of every centroid.
        centroid_movements = np.sum(
            np.abs(new_centroids - centroids) ** p,
            axis=1
        ) ** (1.0 / p)

        centroids = new_centroids

        # Halt only when all centroids changed by at most epsilon.
        if np.all(centroid_movements <= epsilon):
            break

    # Recompute assignments using the final centroids.
    final_distances = l_p_dist_from_centroids(
        X,
        centroids,
        p=p
    )

    cluster_assignments = np.argmin(
        final_distances,
        axis=0
    ).astype(int)

    assigned_distances = final_distances[
        cluster_assignments,
        np.arange(X.shape[0])
    ]

    # Notebook definition:
    # WCSS = sum_i d_Lp(x_i, assigned centroid)^p
    WCSS = float(np.sum(assigned_distances ** p))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, cluster_assignments, WCSS


def norm_pdf(x, mu, sigma):
    """
    Normal density function.
    Inputs:
    - x: a real value or a vector of real values
    - mu: mean of the normal distribution
    - sigma: standard deviation of the normal distribution
    Outputs:
    - prob: the probability densities of the normal distribution at x,
      as a numpy array. If `mu` and `sigma` are scalars, the output shape is
      `(n, 1)`, where n is the number of values in x. If `mu` and `sigma`
      describe k distributions, the output broadcasts to shape `(n, k)`.
    """
    x = np.reshape(x, (-1, 1))
    prob = np.zeros_like(x, dtype=float)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma_eps = 1e-12
    sigma = np.maximum(sigma, sigma_eps)

    prob = (
        1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    ) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

    prob = np.asarray(prob, dtype=float)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return prob


def gmm_pdf(x, weights, mus, sigmas):
    """
    Probability density function of a univariate Gaussian mixture model.
    Inputs:
    - x: a real value or a vector of real values
    - weights: a vector of weights for each Gaussian component
    - mus: a vector of means for each Gaussian component
    - sigmas: a vector of standard deviations for each Gaussian component
    Outputs:
    - prob: the probability densities of the GMM at x,
      as a numpy array of shape `(n, 1)`, where n is the number of values in x.
    """
    x = np.reshape(x, (-1, 1))
    prob = np.zeros_like(x, dtype=float)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    weights = np.asarray(weights, dtype=float).reshape(1, -1)

    component_densities = norm_pdf(
        x,
        mus,
        sigmas
    )

    prob = np.sum(
        component_densities * weights,
        axis=1,
        keepdims=True
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return prob


class GMM(object):
    """
    Fit a Gaussian Mixture Model (EM) to one-dimensional data.

    Parameters
    ----------
    k : int
      Number of Gaussian components
    max_iter : int
      Maximum number of passes over the training dataset in the EM process
    eps : float
      Minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random parameter initialization
    """

    def __init__(self, k=1, max_iter=1000, eps=0.000001, random_state=42):
        """
        THIS CONSTRUCTOR IS ALREADY IMPLEMENTED. DO NOT MODIFY IT.
        """
        # parameters defining the GMM
        self.k = k
        self.weights = None
        self.mus = None
        self.sigmas = None

        # parameters for the EM algorithm
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(self.random_state)

        # attributes for the EM algorithm
        self.responsibilities = None
        self.losses = None

    def get_dist_params(self):
        """
        Return the distribution parameters of the GMM.
        Outputs:
        - params: a dictionary with keys 'weights', 'mus', 'sigmas'
          containing the GMM parameters.
        THIS FUNCTION IS ALREADY IMPLEMENTED. DO NOT MODIFY IT.
        """
        return {
            'weights': self.weights,
            'mus': self.mus,
            'sigmas': self.sigmas
        }

    def init_params(self, X):
        """
        Initialize GMM parameters (weights, mus, sigmas).
        Used in the beginning of the EM algorithm.
        Inputs:
        - X: training data as a 1-D array of shape `(n_examples,)` or a
          2-D column array of shape `(n_examples, 1)`
        THIS FUNCTION IS ALREADY IMPLEMENTED. DO NOT MODIFY IT.
        """
        self.losses = []

        self.weights = np.array([1 / self.k] * self.k)

        if self.k == 1:
            # If k is 1, a good guess is the empirical mean and std.
            self.mus = np.array([np.mean(X)])
            self.sigmas = np.array([np.std(X)])
        else:
            self.mus = np.random.random(self.k)
            self.sigmas = np.random.random(self.k)

            # Avoid starting with sigmas that are too small.
            self.sigmas[self.sigmas < 0.25] += 0.25

    def fit(self, X, verbose=False):
        """
        Fit GMM to data using the EM algorithm.
        Use init_params to initialize all model parameters and then apply the
        EM algorithm by invoking the expectation and maximization functions (see below).
        Store the parameters in attributes of the EM object and the losses in
        self.losses.
        The function halts when the difference between current and previous loss
        is less than eps or when max_iter is reached.
        Inputs:
        - X: training data as a 1-D array of shape `(n_examples,)` or a
          2-D column array of shape `(n_examples, 1)`
        - verbose: if True, print initial parameters and the loss every 5 iterations
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.asarray(X, dtype=float).reshape(-1, 1)

        self.init_params(X)

        if verbose:
            print("Initial parameters:", self.get_dist_params())

        previous_loss = None

        for iteration in range(self.max_iter):
            self.expectation(X)
            self.maximization(X)

            current_loss = float(self.loss(X))
            self.losses.append(current_loss)

            if verbose and iteration % 5 == 0:
                print(
                    "Iteration {}: loss = {}".format(
                        iteration,
                        current_loss
                    )
                )

            if (
                previous_loss is not None
                and abs(previous_loss - current_loss) < self.eps
            ):
                break

            previous_loss = current_loss

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, X):
        """
        Implement the E step of the EM algorithm.
        Calculate the responsibilities (posterior probabilities) of each Gaussian
        component for each data point. Update self.responsibilities.
        Inputs:
        - X: training data as a 1-D array of shape `(n_examples,)` or a
          2-D column array of shape `(n_examples, 1)`
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.asarray(X, dtype=float).reshape(-1, 1)

        weighted_densities = (
            norm_pdf(X, self.mus, self.sigmas)
            * self.weights.reshape(1, -1)
        )

        normalizers = np.sum(
            weighted_densities,
            axis=1,
            keepdims=True
        )

        normalizers = np.maximum(
            normalizers,
            np.finfo(float).tiny
        )

        self.responsibilities = weighted_densities / normalizers

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, X):
        """
        Implement the M step of the EM algorithm.
        Update the GMM parameters (weights, mus, sigmas) based on the current
        responsibilities.
        Inputs:
        - X: training data as a 1-D array of shape `(n_examples,)` or a
          2-D column array of shape `(n_examples, 1)`
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.asarray(X, dtype=float).reshape(-1, 1)

        n_examples = X.shape[0]

        component_sums = np.sum(
            self.responsibilities,
            axis=0
        )

        safe_component_sums = np.maximum(
            component_sums,
            np.finfo(float).eps
        )

        # Update mixture weights.
        self.weights = component_sums / n_examples

        # Update means.
        self.mus = np.sum(
            self.responsibilities * X,
            axis=0
        ) / safe_component_sums

        # Update standard deviations.
        differences = X - self.mus.reshape(1, -1)

        variances = np.sum(
            self.responsibilities * differences ** 2,
            axis=0
        ) / safe_component_sums

        self.sigmas = np.sqrt(variances)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def loss(self, X):
        """
        Calculate the loss function for the GMM.
        The loss is the negative log likelihood of the data given the GMM parameters.
        Inputs:
        - X: training data as a 1-D array of shape `(n_examples,)` or a
          2-D column array of shape `(n_examples, 1)`
        Outputs:
        - c: the loss value (scalar)
        """
        # Truncate very small sigmas to avoid numerical issues.
        sigma_eps = 0.000000000001
        self.sigmas[self.sigmas < sigma_eps] = sigma_eps
        c = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        probabilities = gmm_pdf(
            X,
            self.weights,
            self.mus,
            self.sigmas
        )

        probabilities = np.maximum(
            probabilities,
            np.finfo(float).tiny
        )

        c = float(
            -np.sum(np.log(probabilities))
        )

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return c

    def pdf(self, x):
        """
        Return the probability density function of the GMM at point x.
        Inputs:
        - x: a real value or a vector of real values
        Outputs:
        - prob: the probability densities of the GMM at x,
          as a numpy array of shape `(n, 1)`, where n is the number of values in x.
        """
        prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prob = gmm_pdf(
            x,
            self.weights,
            self.mus,
            self.sigmas
        )

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prob


class NaiveBayesGMM(object):
    """
    Naive Bayes classifier using Gaussian mixture models.

    Parameters
    ----------
    k : int
      Number of Gaussians in each one-dimensional GMM
    random_state : int
      Random number generator seed for random parameter initialization
    """

    def __init__(self, k=1, random_state=42):
        self.k = k
        self.random_state = random_state

        self.prior = None
        self.gmm_dict = None

    def fit(self, X, y, verbose=False):
        """
        Fit class conditional distributions and prior probabilities.
        A GMM is fitted for each feature of each class using the EM algorithm.
        The fitted GMM objects are stored in a dictionary, where the keys are
        class labels and feature indices. The prior probabilities are stored in
        self.prior.
        Inputs:
        - X: training data (n_examples, n_features)
        - y: class labels as a 1-D array of shape `(n_examples,)` or a
          2-D column array of shape `(n_examples, 1)`
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.prior = np.zeros(n_classes, dtype=float)
        self.gmm_dict = {}

        for class_idx, class_label in enumerate(self.classes_):
            class_mask = y == class_label
            class_data = X[class_mask]

            # Empirical class prior.
            self.prior[class_idx] = np.mean(class_mask)

            # Fit one one-dimensional GMM per feature and class.
            for feature_idx in range(n_features):
                gmm = GMM(
                    k=self.k,
                    random_state=self.random_state
                )

                gmm.fit(
                    class_data[:, feature_idx],
                    verbose=verbose
                )

                self.gmm_dict[
                    (class_label, feature_idx)
                ] = gmm

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels.
        Inputs:
        - X: test data (n_examples, n_features)
        Outputs:
        - class_predictions: predicted class labels (n_examples, 1)

        You may compute the MAP rule using log probabilities instead of
        multiplying probabilities directly. This is numerically more stable and
        gives the same argmax.
        """
        class_predictions = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.asarray(X, dtype=float)

        n_examples, n_features = X.shape
        n_classes = len(self.classes_)

        log_scores = np.zeros(
            (n_examples, n_classes),
            dtype=float
        )

        tiny = np.finfo(float).tiny

        for class_idx, class_label in enumerate(self.classes_):
            # Start with the logarithm of the class prior.
            log_scores[:, class_idx] = np.log(
                max(self.prior[class_idx], tiny)
            )

            # Under the Naive Bayes assumption, add the log-density of
            # every feature conditioned on the current class.
            for feature_idx in range(n_features):
                gmm = self.gmm_dict[
                    (class_label, feature_idx)
                ]

                feature_probabilities = gmm.pdf(
                    X[:, feature_idx]
                ).reshape(-1)

                log_scores[:, class_idx] += np.log(
                    np.maximum(feature_probabilities, tiny)
                )

        predicted_indices = np.argmax(
            log_scores,
            axis=1
        )

        class_predictions = self.classes_[
            predicted_indices
        ].reshape(-1, 1)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return class_predictions