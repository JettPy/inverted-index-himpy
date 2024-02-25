import itertools

import numpy as np



"""
Position Transformer
"""


class PositionSetTransformer:
    def __init__(self, splits, element_ndim=1):
        if not isinstance(splits, int) and not isinstance(splits, (list, tuple)):
            raise Exception("Provided splits type is not supported.")
        self._splits = splits
        if isinstance(splits, int) :
            self._ndim = 1
        elif isinstance(splits, (list, tuple)):
            self._ndim = len(splits)

        self._element_ndim = element_ndim
        self._steps = None
        self._size = None
        self._element_regions = None
        self._element_mapping = None
        self._last_transformed = None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, batch=False, ids=None):
        if batch is True:
            masks = list()
            X_ = X if ids is None else X[ids]
            for item in X_:
                size = item.shape[:-1] if self._element_ndim > 1 else item.shape
                # Avoid recalculation elements if a previous fit call had the same size
                if self._size is None or size != self._size:
                    self._size = size
                    self._last_transformed = None
                    self._compose_low_elements()
                    masks.append(self._transform())
                else:
                    masks.append(self._last_transformed)
            return masks
        else:
            size = X.shape[:-1] if self._element_ndim > 1 else X.shape
            # Avoid recalculation elements if a previous fit call had the same size
            if self._size is None or size != self._size:
                self._size = size
                self._last_transformed = None
                self._compose_low_elements()
                return self._transform()
            return self._last_transformed

    def fit_transform(self, X=None, y=None, batch=False, ids=None):
        return self.fit(X, y).transform(X, batch, ids=ids)

    def filter_elements(self, X, element_ids=None):
        if not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if element_ids is None:
            return X
        if len(X.shape) == 2:
            mask = np.zeros(X.shape, dtype=np.uint8)
            filtering_elements = np.array(tuple(element_ids), dtype=np.uint8)
            indices = np.nonzero(np.isin(X, filtering_elements))
            mask[indices] = X[indices]
            return mask

    # def filter_elements(self, element_ids):
    #     pos_mask = np.zeros(self._size, dtype=np.int)
    #     if isinstance(element_ids, (str, int)):
    #         element_ = self.find_element(element_ids)
    #         for low_element in itertools.product(
    #                 *[range(element_[i], element_[self._ndim + i] + 1) for i in range(self._ndim)]
    #         ):
    #             pos_mask[low_element] = element_ids
    #         return pos_mask
    #
    #     if isinstance(element_ids, (tuple, list, set)):
    #         for element_id in element_ids:
    #             element_ = self.find_element(element_id)
    #             for low_element in itertools.product(
    #                 *[range(element_[i], element_[self._ndim+i]+1) for i in range(self._ndim)]
    #             ):
    #                 pos_mask[low_element] = element_id
    #         return pos_mask

    def filter_data(self, X, I, element_ids=None):
        if element_ids and not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if len(X.shape) == 2:
            mask = np.full(I.shape, fill_value=255, dtype=np.uint8)
            if element_ids is None:
                indices = np.nonzero(X)
                mask[indices] = I[indices]
            else:
                filtering_elements = np.array(tuple(element_ids), dtype=np.uint8)
                indices = np.nonzero(np.isin(X, filtering_elements))
                mask[indices] = I[indices]
            return mask

    # def filter_data(self, X, I, element_ids):
    #     mask = np.full(data.shape, fill_value=0, dtype=np.int)
    #     if isinstance(element_ids, (str, int)):
    #         element_region = self.find_element(element_ids)
    #         for low_element in self._mask_element(element_region):
    #             mask[low_element] = data[low_element]
    #         return mask
    #     if isinstance(element_ids, (tuple, list, set)):
    #         for element_id in element_ids:
    #             element_region = self.find_element(element_id)
    #             for low_element in self._mask_element(element_region):
    #                 mask[low_element] = data[low_element]
    #         return mask

    # TODO: make return of find_element and find_elements in same manner
    def find_element(self, element_id):
        if element_id is not None and element_id != "":
            if element_id in self._element_mapping:
                return self._element_regions[self._element_mapping[element_id]]
        return None

    def find_elements(self, element_ids):
        if element_ids is not None and element_ids != "":
            for element_id in element_ids:
                if element_id in self._element_mapping:
                    yield element_id, self._element_regions[self._element_mapping[element_id]]
        else:
            for element in self:
                yield element

    def _transform(self):
        mask = np.zeros(self._size, dtype=np.uint8)
        for element_id, element_region in self:
            for low_element in itertools.product(
                *[range(element_region[i], element_region[self._ndim+i]+1) for i in range(self._ndim)]
            ):
                mask[low_element] = element_id
        self._last_transformed = mask
        return mask

    def _compose_low_elements(self):
        if isinstance(self._size, int):
            step = self._size//self._splits
            abs_intervals = self._build_absolute_intervals(step, self._splits)
            self._element_regions = list()
            for start in abs_intervals:
                self._element_regions.append((start, start + step - 1))
            self._build_position_elements()
        elif isinstance(self._size, (list, tuple)):
            steps = list()
            abs_intervals = list()
            for i, dim_size in enumerate(self._size):
                steps.append(dim_size // self._splits[i])
                abs_intervals.append(self._build_absolute_intervals(steps[i], self._splits[i]))
            self._element_regions = list()
            for start in itertools.product(*abs_intervals):
                grid_start = list()
                grid_end = list()
                for i, start_dim_item in enumerate(start):
                    grid_start.append(start_dim_item)
                    grid_end.append(start_dim_item + steps[i] - 1)
                self._element_regions.append(tuple(grid_start + grid_end))
            self._build_position_elements()
        else:
            raise Exception("Provided unsupported type.")

    def _build_absolute_intervals(self, dim_step, dim_splits):
        return [i*dim_step for i in range(dim_splits)]

    def _build_position_elements(self):
        self._element_mapping = dict()
        for i in range(len(self._element_regions)):
            self._element_mapping[str(i + 1)] = i

    def _mask_element(self, element_region):
        for low_elements in itertools.product(
            *[range(element_region[i], element_region[self._ndim+i]+1) for i in range(self._ndim)]
        ):
            yield low_elements

    def __iter__(self):
        for key, value in iter(self._element_mapping.items()):
            yield key, self._element_regions[value]