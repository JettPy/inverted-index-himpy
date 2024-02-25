import colorsys

import numpy as np


"""
Color Transformer
"""


COLOR_ELEMENTS = ({"id": "e1", "h": [40, 54], "s": [30, 79], "b": [20, 59]}, {"id": "e2", "h": [40, 54], "s": [80, 159], "b": [20, 59]}, {"id": "e3", "h": [40, 54], "s": [160, 240], "b": [20, 59]}, {"id": "e4", "h": [55, 104], "s": [0, 29], "b": [60, 159]}, {"id": "e5", "h": [55, 104], "s": [30, 79], "b": [20, 59]}, {"id": "e6", "h": [55, 104], "s": [30, 79], "b": [60, 159]}, {"id": "e7", "h": [55, 104], "s": [30, 79], "b": [160, 219]}, {"id": "e8", "h": [55, 104], "s": [30, 79], "b": [220, 240]}, {"id": "e9", "h": [55, 104], "s": [80, 159], "b": [20, 59]}, {"id": "e10", "h": [55, 104], "s": [80, 159], "b": [60, 159]}, {"id": "e11", "h": [55, 104], "s": [80, 159], "b": [160, 219]}, {"id": "e12", "h": [55, 104], "s": [80, 159], "b": [220, 240]}, {"id": "e13", "h": [55, 104], "s": [160, 240], "b": [20, 59]}, {"id": "e14", "h": [55, 104], "s": [160, 240], "b": [60, 159]}, {"id": "e15", "h": [55, 104], "s": [160, 240], "b": [160, 219]}, {"id": "e16", "h": [55, 104], "s": [160, 240], "b": [220, 240]}, {"id": "e17", "h": [105, 114], "s": [30, 79], "b": [60, 159]}, {"id": "e18", "h": [105, 114], "s": [80, 159], "b": [60, 159]}, {"id": "e19", "h": [105, 114], "s": [160, 240], "b": [20, 59]}, {"id": "e20", "h": [105, 114], "s": [160, 240], "b": [60, 159]}, {"id": "e21", "h": [30, 39], "s": [30, 79], "b": [220, 240]}, {"id": "e22", "h": [40, 54], "s": [30, 79], "b": [60, 159]}, {"id": "e23", "h": [40, 54], "s": [30, 79], "b": [160, 219]}, {"id": "e24", "h": [40, 54], "s": [30, 79], "b": [220, 240]}, {"id": "e25", "h": [40, 54], "s": [80, 159], "b": [60, 159]}, {"id": "e26", "h": [40, 54], "s": [80, 159], "b": [160, 219]}, {"id": "e27", "h": [40, 54], "s": [80, 159], "b": [220, 240]}, {"id": "e28", "h": [43, 54], "s": [160, 240], "b": [60, 159]}, {"id": "e29", "h": [40, 54], "s": [160, 240], "b": [220, 240]}, {"id": "e30", "h": [45, 54], "s": [160, 240], "b": [160, 219]}, {"id": "e31", "h": [225, 9], "s": [30, 79], "b": [60, 159]}, {"id": "e32", "h": [225, 9], "s": [30, 79], "b": [160, 219]}, {"id": "e33", "h": [225, 9], "s": [80, 159], "b": [20, 59]}, {"id": "e34", "h": [225, 9], "s": [80, 159], "b": [60, 159]}, {"id": "e35", "h": [225, 9], "s": [80, 159], "b": [160, 219]}, {"id": "e36", "h": [225, 9], "s": [80, 159], "b": [220, 240]}, {"id": "e37", "h": [225, 9], "s": [160, 240], "b": [20, 59]}, {"id": "e38", "h": [225, 9], "s": [160, 240], "b": [60, 159]}, {"id": "e39", "h": [225, 9], "s": [160, 240], "b": [160, 219]}, {"id": "e40", "h": [225, 9], "s": [160, 240], "b": [220, 240]})


def hsl2rgb(h, s, b):
    r, g, b = colorsys.hls_to_rgb(h/240, b/240, s/240)
    return round(255 * r), round(255 * g), round(255 * b)


def rgb2hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h * 240, s * 240, l * 240


def get_h_avg(h):
    return 0.0 if h[0] > h[1] else h[0] + (h[1] - h[0]) / 2


COLOR_ELEMENTS_RGB = {
    el["id"]: hsl2rgb(
        get_h_avg(el["h"]),
        el["s"][0] + (el["s"][1] - el["s"][0]) / 2,
        el["b"][0] + (el["b"][1] - el["b"][0]) / 2)
    for el in COLOR_ELEMENTS
}


class ColorSetTransformer:
    """

    FIXME: inner_ids vs (element_id, element_name)?

    """

    def __init__(self, elements=COLOR_ELEMENTS):

        self._mapping_inner_id = None
        self._mapping_feature = None
        # self._mapping_display_color = None
        self._element_dtype = np.uint8
        self._compose_low_elements(elements)
        # self._assign_display_colors()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, batch=False):
        if batch is True:
            raise NotImplementedError
        else:
            X_flatten = X.reshape((-1, X.shape[-1]))
            mask = np.zeros(X_flatten.shape[0], dtype=self._element_dtype)
            for i in range(len(X_flatten)):
                mask[i] = self._convert2element(*rgb2hsl(*X_flatten[i]))
            return mask.reshape(X.shape[:-1])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def filter_elements(self, X, element_ids, batch=False):
        """
        Filter elements in provided data.

        Parameters
        ----------
        X : ndarray
            Transformed data that uses inner ids as elements.
        element_ids : list, tuple, set
            Element ids to filter.
        single : bool
            If true (default), then provided X is one object data. Otherwise,
            X contains multiple objects

        Returns
        -------
        mask : ndarray
            Data that contains only elements from element_ids
        """
        mask = np.zeros(X.shape, dtype=self._element_dtype)
        if batch is True:
            raise NotImplementedError
        else:
            for element_id in element_ids:
                indices = X==element_id
                mask[indices] = X[indices]
            return mask

    def filter_data(self, X, I, element_ids=None):
        if element_ids and not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if len(X.shape) == 2:
            mask = np.full(I.shape, fill_value=255, dtype=np.uint8)
            if element_ids is None:
                indices = np.nonzero(X)
                mask[indices] = I[indices]
            else:
                filtering_elements = np.array(tuple(element_ids))
                indices = np.nonzero(np.isin(X, filtering_elements))
                mask[indices] = I[indices]
            return mask

    # def filter_data(self, X, I, element_ids, batch=False):
    #     if batch:
    #         raise NotImplementedError
    #     else:
    #         X_flatten = X.reshape((-1, X.shape[-1]))
    #         mask = np.zeros(X_flatten.shape, dtype=X.dtype)
    #         if isinstance(element_ids, (str, int)):
    #             for i in range(len(X_flatten)):
    #                 element_id = self._convert2element(*rgb2hsl(*X_flatten[i]))
    #                 if element_id == element_ids:
    #                     mask[i,:] = X_flatten[i,:]
    #             return mask.reshape(X.shape)
    #         if isinstance(element_ids, (tuple, list, set)):
    #             for i in range(len(X_flatten)):
    #                 element_id = self._convert2element(*rgb2hsl(*X_flatten[i]))
    #                 if element_id in element_ids:
    #                     mask[i,:] = X_flatten[i,:]
    #             return mask.reshape(X.shape)
    #     raise NotImplementedError

    def transform_to_int(self, X, is_batch=False):
        if is_batch is True:
            pass
        else:
            X_flatten = X.reshape(-1)
            convert = np.vectorize(
                lambda x: self._mapping_inner_id[x] if x in self._mapping_inner_id else 0,
                otypes=(np.uint8,))
            return convert(X_flatten).reshape(X.shape)

    def _compose_low_elements(self, elements):
        self._mapping_feature = dict()
        self._mapping_inner_id = dict()
        if isinstance(elements[0]["id"], str):
            self._element_dtype = "<U10"
        for i, item in enumerate(elements):
            self._mapping_feature[item["id"]] = item
            self._mapping_inner_id[item["id"]] = i+1

    def _convert2element(self, h, s, l):
        for el_id, el in self._mapping_feature.items():
            h_cond = el["h"][0] <= h <= el["h"][1] if el["h"][0] <= el["h"][1] \
                else el["h"][0] <= h <= 240 or 0 <= h <= el["h"][1]
            if h_cond and el["s"][0] <= s <= el["s"][1] and el["b"][0] <= l <= el["b"][1]:
                return el["id"]
