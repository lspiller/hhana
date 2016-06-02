import numpy as np
import scipy.sparse as sp
import numbers
import warnings

def check_cv(cv, X=None, y=None, classifier=False):
    """Input checker utility for building a CV in a user friendly way.

    Parameters
    ----------
    cv : int, a cv generator instance, or None
        The input specifying which cv generator to use. It can be an
        integer, in which case it is the number of folds in a KFold,
        None, in which case 3 fold is used, or another object, that
        will then be used as a cv generator.

    X : array-like
        The data the cross-val object will be applied on.

    y : array-like
        The target variable for a supervised learning problem.

    classifier : boolean optional
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv: a cross-validation generator instance.
        The return value is guaranteed to be a cv generator instance, whatever
        the input type.
    """
    return _check_cv(cv, X=X, y=y, classifier=classifier, warn_mask=True)


def _check_cv(cv, X=None, y=None, classifier=False, warn_mask=False):
    # This exists for internal use while indices is being deprecated.
    is_sparse = sp.issparse(X)
    needs_indices = is_sparse or not hasattr(X, "shape")
    if cv is None:
        cv = 3
    if isinstance(cv, numbers.Integral):
        if warn_mask and not needs_indices:
            warnings.warn('check_cv will return indices instead of boolean '
                          'masks from 0.17', DeprecationWarning)
        else:
            needs_indices = None
        if classifier:
            cv = StratifiedKFold(y, cv, indices=needs_indices)
        else:
            if not is_sparse:
                n_samples = len(X)
            else:
                n_samples = X.shape[0]
            cv = KFold(n_samples, cv, indices=needs_indices)
    if needs_indices and not getattr(cv, "_indices", True):
        raise ValueError("Sparse data and lists require indices-based cross"
                         " validation generator, got: %r", cv)
    return cv
def safe_mask(X, mask):
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask: array
        Mask to be used on X.

    Returns
    -------
        mask
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.int):
        return mask

    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask

def _safe_split(estimator, X, y, sample_weight, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels."""
    if hasattr(estimator, 'kernel') and callable(estimator.kernel):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_subset = [X[idx] for idx in indices]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            if train_indices is None:
                X_subset = X[np.ix_(indices, indices)]
            else:
                X_subset = X[np.ix_(indices, train_indices)]
        else:
            X_subset = X[safe_mask(X, indices)]

    if y is not None:
        y_subset = y[safe_mask(y, indices)]
    else:
        y_subset = None

    if sample_weight is not None:
        sample_weight_subset = np.asarray(sample_weight)[indices]
    else:
        sample_weight_subset = None

    return X_subset, y_subset, sample_weight_subset

