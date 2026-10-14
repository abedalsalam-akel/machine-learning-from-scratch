# Machine Learning From Scratch

Python and NumPy implementations adapted from six submitted **Machine Learning from Data** course assignments at Reichman University. Organized by homework to preserve the original interfaces and provenance; this is **not** a production-ready ML library.

## Implementations

| Module | Coursework algorithms |
| --- | --- |
| `hw1.py` | Linear regression, gradient descent, pseudoinverse, LASSO, K-nearest neighbors |
| `hw2.py` | Decision trees, entropy/Gini, pruning and cross-validation |
| `hw3.py` | Logistic regression, mini-batch gradient descent, classification metrics, one-vs-all |
| `hw4.py` | Linear and kernel SVMs, hinge loss, cross-validation |
| `hw5.py` | Poisson likelihood and MLE, confidence intervals, Naive Bayes |
| `hw6.py` | K-means, Gaussian mixtures, EM and GMM-based classification |

## Run tests

Python 3.10+:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

These are selected **smoke tests**, not comprehensive validations of every algorithm or numerical edge case. Advanced numerical routines need further testing before use on real datasets.

## Attribution and publication

Adapted from university course submissions for portfolio presentation. Original starter scaffolding and API specifications may belong to course staff. Attribution, collaborators' contributions, and university publication permissions must be confirmed before this repository is made public.

No deployed ML service, benchmark accuracy, or general-purpose library is claimed. No license is granted.
