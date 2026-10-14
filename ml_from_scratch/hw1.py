###### Your ID ######
# Identifier removed for portfolio publication
#####################

# imports
import numpy as np


def add_bias_feature(X):
    """
    Adds a bias feature to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    ###########################################################################
    # TODO: Implement the function by adding a column of ones to the data.   #
    ###########################################################################
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack((np.ones(X.shape[0]), X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def compute_mean_squared_error(X, y, w):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    The loss is J = (1/n) * ||Xw - y||^2 

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - w: the parameters (weights) of the model being learned.

    Returns:
    - J: the mean squared error loss associated with the current set of 
         parameters (single number).
    """
    ###########################################################################
    # TODO: Implement the MSE loss function.                                  #
    ###########################################################################
    errors = X @ w - y
    return np.mean(errors ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def gradient_descent(X, y, w, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent on the training
    set.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - w: Initial values of the parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - w_star: The learned parameters of your model.
    - J_history: the loss value after every iteration (not inlcuding before the 1st iteration).
    """
    w = w.copy()  # optional: w outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the simple gradient descent optimization algorithm.     #
    ###########################################################################
    n = X.shape[0]

    for _ in range(num_iters):
        grad = (2 / n) * (X.T @ (X @ w - y))
        w = w - eta * grad
        J_history.append(compute_mean_squared_error(X, y, w))

    return w, J_history
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def gradient_descent_stop_condition(X, y, w, eta, max_iter, epsilon=1e-5):
    """
    Learn the parameters of your model using the training set, but stop the
    learning process once the gradient norm is smaller than epsilon.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - w: Initial values of the parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the gradient norm (stop when ||gradient|| < epsilon).

    Returns:
    - w_star: The learned parameters of your model.
    - J_history: the loss value for every iteration (not inlcuding before the 1st iteration).
    """
    w = w.copy()  # optional: w outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement gradient descent with gradient-norm stopping condition. #
    ###########################################################################
    n = X.shape[0]

    for _ in range(max_iter):
        grad = (2 / n) * (X.T @ (X @ w - y))
        if np.linalg.norm(grad) < epsilon:
            break
        w = w - eta * grad
        J_history.append(compute_mean_squared_error(X, y, w))

    return w, J_history
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def find_best_learning_rate(X, y, w_init, iterations, eta_values):
    """
    Iterate over the provided values of eta and train a multivariate linear
    regression model using the training dataset. Maintain a python dictionary
    with eta as the key and the loss as the value.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - w_init: Initial values of the parameters (weights) of the model being learned.
    - iterations: the number of updates to run for each eta
    - eta_values: The list of eta values to consider
    Returns:
    - eta_dict: A python dictionary - {eta_value: loss history}
    """
    
    history_per_eta = {}  # {eta_value: loss_history}
    ###########################################################################
    # TODO: Implement the function that runs GD with multiple etas.           #
    ###########################################################################
    for eta in eta_values:
        _, J_history = gradient_descent(X, y, w_init.copy(), eta, iterations)
        history_per_eta[eta] = J_history

    return history_per_eta
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - w_star: The optimal parameters of your model.
    """
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    return np.linalg.inv(X.T @ X) @ X.T @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

def mini_batch_gradient_descent(X, y, w, eta, num_epochs, batch_size):
    """
    Learn the parameters of the model using mini-batch gradient descent.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - w: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_epochs: The number of passes over the training data.
    - batch_size: The number of samples in each mini-batch.

    Returns:
    - w_star: The learned parameters of your model.
    - J_history: the loss values recorded after each *epoch* of training (not inlcuding before the 1st epoch).
    """
    w = w.copy()
    J_history = []
    np.random.seed(1)
    ###########################################################################
    # TODO: Implement mini-batch gradient descent.                            #
    # recall to shuffle the data before each epoch                            #
    # you can use the following three lines to shuffle the data efficiently:  #
    #   shuffled_indices = np.random.permutation(X.shape[0])                  #
    #   X_shuffled = X[shuffled_indices]                                      #
    #   y_shuffled = y[shuffled_indices]                                      #
    ###########################################################################
    n = X.shape[0]

    for _ in range(num_epochs):
        shuffled_indices = np.random.permutation(X.shape[0])
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            batch_n = X_batch.shape[0]
            grad = (2 / batch_n) * (X_batch.T @ (X_batch @ w - y_batch))
            w = w - eta * grad

        J_history.append(compute_mean_squared_error(X, y, w))

    return w, J_history
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def solve_LASSO(X, y, lmda, w_init=None, eta=0.01, max_iter=10_000):
    """
    Solve LASSO regression (L1-regularized least squares).

    Minimizes the objective used in the assignment (Part 2, regularization section):
        (1/n) * ||Xw - y||^2 + lambda * ||w||_1
    i.e. mean squared error on the squared residuals plus an L1 penalty on all weights.
    Use (sub-)gradient descent.

    Input:
    - X: Input data with bias term (n instances over p+1 features).
    - y: True labels (n instances).
    - lmda: The LASSO regularization hyperparameter (lambda).
    - w_init: Initial weights, or None to start from the zero vector.
    - eta: The learning rate of your model.
    - max_iter: The number of gradient updates to perform.

    Returns:
    - w_star: The learned parameters of your model.
    """

    if w_init is None:
        w = np.zeros(X.shape[1])
    else:
        w = w_init.copy()

    ###########################################################################
    #    Your code here
    ###########################################################################
    n = X.shape[0]

    for _ in range(max_iter):
        mse_grad = (2 / n) * (X.T @ (X @ w - y))
        subgrad_l1 = lmda * np.sign(w)
        subgrad_l1[w == 0] = 0
        grad = mse_grad + subgrad_l1
        w = w - eta * grad

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return w


def choose_best_L1_parameter(X_train, y_train, X_val, y_val, lambda_values):
    """
    Find the best L1 regularization parameter using a validation set.

    Input:
    - X_train, y_train: the training data (X has bias column)
    - X_val, y_val: the validation data (X has bias column)
    - lambda_values: list of lambda values to try

    Returns:
    - best_lambda: The lambda that minimizes validation loss.
    - results_dict: A dictionary {lambda_value: {\"train_loss\": float, \"val_loss\": float}}
    """
    best_lambda = None
    results_dict = {}
    ###########################################################################
    # TODO: Implement the function. For each lambda, fit LASSO regression    #
    # on train, record train and validation MSE loss. Return best lambda and dict. #
    ###########################################################################
    best_val_loss = np.inf

    for lmda in lambda_values:
        w_lasso = solve_LASSO(X_train, y_train, lmda)
        train_loss = compute_mean_squared_error(X_train, y_train, w_lasso)
        val_loss = compute_mean_squared_error(X_val, y_val, w_lasso)

        results_dict[lmda] = {"train_loss": train_loss, "val_loss": val_loss}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lambda = lmda

    return best_lambda, results_dict
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def knn_predict(X_train, y_train, X_test, k):
    """
    Predict target values for the test data using K-nearest neighbors
    regression with Euclidean distance.

    Input:
    - X_train: Training data (n_train instances over p features).
    - y_train: Training labels (n_train instances).
    - X_test: Test data (n_test instances over p features).
    - k: Number of neighbors to use.

    Returns:
    - y_pred: Predicted values for the test data (n_test instances).
    """
    y_pred = []
    ###########################################################################
    # TODO: Implement K-nearest neighbors regression.                        #
    ###########################################################################
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        knn_indices = np.argsort(distances)[:k]
        y_pred.append(np.mean(y_train[knn_indices]))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return y_pred