# Author: Sergei Papulin <papulin.edu@gmail.com>

import numpy as np
from himpy.histogram import Histogram1D, HElement, Histogram, HElementSet


##################
#
# COMMON
#
##################


"""
Feature Merger
"""


# TODO: Check whether it is used anywhere
# def merge_features(*features):
#     features_ = None
#     shape = features[0].shape
#     ndim_features = len(features)
#     features_flatten = tuple([item.reshape(-1) for item in features])
#     if ndim_features == 1:
#         features_ = np.rec.fromarrays(features_flatten[0])
#     elif ndim_features > 1:
#         features_ = np.rec.fromarrays(features_flatten)
#     return features_.reshape(shape)


class FeatureMerger:
    def __init__(self):
        self._ndim = None

    def fit(self, X, batch=False):
        self._ndim = len(X[0]) if batch else len(X)
        return self

    def fit_transform(self, X, batch=False):
        return self.fit(X).transform(X, batch=batch)

    def transform(self, X, batch=False):

        if self._ndim is None:
            raise Exception("Use the fit method at first.")

        ndim = len(X[0]) if batch else len(X)
        if self._ndim != ndim:
            raise Exception("Mismatching dimensions.")

        if batch is True:
            merged_features = list()
            for features in range(zip(*X)):
                merged_features.append(self._merge_single(features))
            return merged_features
        else:
            return self._merge_single(X)

    def _merge_single(self, X):
        features_ = None
        shape = X[0].shape
        features_flatten = tuple([item.reshape(-1) for item in X])
        if self._ndim == 1:
            features_ = features_flatten[0]
        elif self._ndim > 1:
            features_ = np.rec.fromarrays(features_flatten)
        return features_.reshape(shape)


"""
Histogram Creator
"""


# TODO: Combine create_histogram and create_histogram_
def create_histogram(features, normalize=True):
    """
    Faster than create_histogram_
    """
    features_ = None
    ndim_features = len(features)
    features_flatten = tuple([item.reshape(-1) for item in features])
    if ndim_features == 1:
        features_ = features_flatten[0]
    elif ndim_features > 1:
        # None: if some feature is str, then all features_ will be str
        features_ = np.c_[features_flatten]
        # preserve only items with non-zero element
        # Note: zero element means NaN as element ids is not equal to 0
        zero_ = 0 if np.issubdtype(features_.dtype, np.integer) else "0"
        features_ = features_[np.all(features_ != zero_, axis=1)]
        # for i in range(features_.shape[1]):
        #     features_ = features_[features_[:,i]!=0]

    num_elements = features_flatten[0].shape[0]
    elements, counts = np.unique(features_, axis=0, return_counts=True)
    if normalize:
        counts = counts / num_elements
    hist = Histogram1D(data=None)
    for i in range(len(counts)):
        item_tuple = elements[i]
        if ndim_features > 1:
            item_tuple = tuple(map(str, elements[i].tolist()))
        hist[item_tuple] = HElement(item_tuple, counts[i])
    return hist


def create_histogram_(merged_features, normalize=True):
    """
    Slower than create_histogram?
    """
    features_ = merged_features.reshape(-1)
    ndim_element = len(features_[0]) if isinstance(features_[0], np.record) else 1

    if ndim_element > 1:
        # preserve only items with non-zero element
        # Note: zero element means NaN as element ids is not equal to 0
        for name in features_.dtype.names:
            # check type str or int to use appropriate zero
            zero_ = 0 if np.issubdtype(features_[name].dtype, np.integer) else "0"
            features_ = features_[features_[name] != zero_]

    elements, counts = np.unique(features_, axis=0, return_counts=True)
    if normalize:
        counts = counts / merged_features.size
    hist = Histogram1D(data=None)
    for i in range(len(counts)):
        item_tuple = elements[i]
        if ndim_element > 1:
            item_tuple = tuple(map(str, elements[i].tolist()))
        hist[item_tuple] = HElement(item_tuple, counts[i])
    return hist


def create_histogram_batch_(X, normalize=True):
    hists = list()
    for i in range(len(X)):
        hists.append(create_histogram_(X[i], normalize=normalize))
    return hists


# TODO: move to ???
def convert_complete_histogram_values(U, H, to_sort=False):
    """
    Convert a sparse form of a histogram to a list of values of all elements.
    Absent elements in the sparse form will be replaced by zero values.

    Parameters
    ----------
    U - the universal set of elements
    H - histogram of data or element
    to_sort - whether it's needed to sort elements by names

    Returns
    -------
    Values of all elements
    """
    hist_val_full = [0 for _ in U]

    if isinstance(H, Histogram):
        for i in range(len(U)):
            if U[i] in H:
                hist_val_full[i] = H[U[i]].value
    elif isinstance(H, HElementSet):
        elements = H.to_dict()
        for i in range(len(U)):
            if U[i] in elements:
                hist_val_full[i] = elements[U[i]]

    if to_sort:
        hist_val_full.sort(reverse=False)

    return hist_val_full


"""
Filters 
"""


def filter_data(data, features, elements):

    # TODO: check?
    # def to_int(items):
    #     return map(int, items)

    features_ = features.reshape(-1)
    if len(data.shape) == len(features.shape) + 1:
        data_flatten = data.reshape(-1, data.shape[-1])
    elif len(data.shape) == len(features.shape):
        data_flatten = data.reshape(-1)
    else:
        raise Exception()

    mask = np.full(data_flatten.shape, fill_value=255, dtype=int)

    # features_ = np.rec.fromarrays(features_flatten)
    # elements_ = np.array(
    #     list(itertools.product(*(to_int(items) for items in elements))),
    #     dtype=features_.dtype)

    # elements_ = np.array(
    #     list(itertools.product(*(items for items in elements))),
    #     dtype=features_.dtype)

    elements_ = np.array(elements, dtype=features_.dtype)

    indx = np.in1d(features_, elements_)
    mask[indx] = data_flatten[indx]

    return mask.reshape(data.shape)


"""
Element Extractor
"""


# TODO: combine extract_elements and extract_element_set
def extract_elements(data):
    return np.unique(data[data != 0])


def extract_element_set(HE, none_dim=1):
    """

    Parameters
    ----------
    HE : HElementSet
    none_dim : int
        used to return empty dicts if no elements

    Returns
    -------

    """
    elements = list(HE.to_dict().keys())
    if len(elements) == 0 and none_dim > 1:
        return [dict() for _ in range(none_dim)]
    elif len(elements) == 0:
        return None
    dim = len(elements[0])
    result = list()
    for i in range(dim):
        result.append({el[i] for el in elements})
    return result
