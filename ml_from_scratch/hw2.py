
import numpy as np


def calc_gini(data):
    gini = 0.0
    if data is None or len(data) == 0:
        return 0.0

    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1 - np.sum(probabilities ** 2)
    return gini


def calc_entropy(data):
    entropy = 0.0
    if data is None or len(data) == 0:
        return 0.0

    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1, depth=0, max_depth=1000, gain_ratio=False):
        self.data = data
        self.terminal = False
        self.feature = feature
        self.pred = self.calc_node_pred()
        self.depth = depth
        self.children = []
        self.children_values = []
        self.max_depth = max_depth
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        labels = self.data[:, -1]
        values, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)
        candidates = values[counts == max_count]
        pred = np.min(candidates)
        return pred

    def add_child(self, node, val):
        self.children.append(node)
        self.children_values.append(val)

    def goodness_of_split(self, feature):
        goodness = 0
        groups = {}

        values = np.unique(self.data[:, feature])
        n = len(self.data)

        for val in values:
            groups[val] = self.data[self.data[:, feature] == val]

        if self.gain_ratio:
            parent_impurity = calc_entropy(self.data)
        else:
            parent_impurity = self.impurity_func(self.data)

        weighted_impurity = 0.0
        split_info = 0.0

        for val in values:
            subset = groups[val]
            weight = len(subset) / n

            if self.gain_ratio:
                weighted_impurity += weight * calc_entropy(subset)
                if weight > 0:
                    split_info -= weight * np.log2(weight)
            else:
                weighted_impurity += weight * self.impurity_func(subset)

        impurity_reduction = parent_impurity - weighted_impurity

        if self.gain_ratio:
            if split_info == 0:
                goodness = 0.0
            else:
                goodness = impurity_reduction / split_info
        else:
            goodness = impurity_reduction

        return goodness, groups

    def split(self):
        labels = self.data[:, -1]

        if len(np.unique(labels)) == 1:
            self.terminal = True
            return

        if self.depth >= self.max_depth:
            self.terminal = True
            return

        num_features = self.data.shape[1] - 1
        best_feature = None
        best_goodness = -np.inf
        best_groups = None

        for feature in range(num_features):
            goodness, groups = self.goodness_of_split(feature)

            if best_feature is None or goodness > best_goodness or (
                goodness == best_goodness and feature < best_feature
            ):
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

        if best_feature is None or best_goodness <= 0:
            self.terminal = True
            return

        self.feature = best_feature

        for val in np.unique(self.data[:, best_feature]):
            child = DecisionNode(
                data=best_groups[val],
                impurity_func=self.impurity_func,
                feature=-1,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            self.add_child(child, val)
            child.split()


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, max_depth=1000, gain_ratio=False):
        self.data = data
        self.root = None
        self.max_depth = max_depth
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio

    def depth(self):
        return self.root.depth

    def build_tree(self):
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            feature=-1,
            depth=0,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )
        self.root.split()

    def predict(self, instance):
        node = self.root

        while node is not None and not node.terminal:
            feature_value = instance[node.feature]

            if feature_value in node.children_values:
                child_index = node.children_values.index(feature_value)
                node = node.children[child_index]
            else:
                return node.pred

        return node.pred

    def calc_accuracy(self, dataset):
        if dataset is None or len(dataset) == 0:
            return 0

        correct = 0
        for instance in dataset:
            if self.predict(instance) == instance[-1]:
                correct += 1

        accuracy = correct / len(dataset)
        return accuracy


def depth_pruning(X_train, X_validation):
    training = []
    validation = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(
            data=X_train,
            impurity_func=calc_entropy,
            max_depth=max_depth,
            gain_ratio=True
        )
        tree.build_tree()

        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))

    return training, validation


def cross_validate_depth(X_train, num_folds=5):
    best_depth = None
    cv_acc = []

    folds = np.array_split(X_train, num_folds)
    best_accuracy = -np.inf

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        fold_accuracies = []

        for fold_index in range(num_folds):
            validation_fold = folds[fold_index]
            train_folds = [folds[i] for i in range(num_folds) if i != fold_index]
            train_data = np.concatenate(train_folds, axis=0)

            tree = DecisionTree(
                data=train_data,
                impurity_func=calc_entropy,
                max_depth=max_depth,
                gain_ratio=True
            )
            tree.build_tree()

            fold_accuracies.append(tree.calc_accuracy(validation_fold))

        mean_accuracy = np.mean(fold_accuracies)
        cv_acc.append(mean_accuracy)

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_depth = max_depth

    return best_depth, cv_acc


def build_best_tree(X_train, X_test, num_folds=5):
    final_tree = None
    best_depth = None
    train_acc = 0
    test_acc = 0

    best_depth, _ = cross_validate_depth(X_train, num_folds=num_folds)

    final_tree = DecisionTree(
        data=X_train,
        impurity_func=calc_entropy,
        max_depth=best_depth,
        gain_ratio=True
    )
    final_tree.build_tree()

    train_acc = final_tree.calc_accuracy(X_train)
    test_acc = final_tree.calc_accuracy(X_test)

    return final_tree, best_depth, train_acc, test_acc