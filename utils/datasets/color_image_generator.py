import copy
import random
import numpy as np

from ..feature_extraction.color import COLOR_ELEMENTS_RGB, hsl2rgb


class ColorImageGenerator:

    def __init__(self, elements=None, element_type="hsb40"):
        self._elements = None
        if element_type == "hsb40":
            self._elements = copy.deepcopy(COLOR_ELEMENTS_RGB)
        elif elements is not None:
            self._elements = copy.deepcopy(elements)
        else:
            raise AttributeError

    def generate(self, shape, element_ids="all", steps=(5, 5), normal_element_ids=None, random_state=None):

        random.seed(random_state)

        num_splits_x = shape[1] // steps[1]
        num_splits_y = shape[0] // steps[0]

        element_ids_ = list(self._elements.keys()) if element_ids == "all" else element_ids

        image = np.zeros(list(shape) + [3], dtype=np.uint8)

        for i in range(num_splits_y):
            for j in range(num_splits_x):
                image[i*steps[1]:(i+1)*steps[1], j*steps[0]:(j+1)*steps[0], :] = \
                    self._elements[random.choice(list(element_ids_))]

        if not normal_element_ids:
            return image

        cx = random.randint(0, num_splits_x)
        cy = random.randint(0, num_splits_y)

        sx = num_splits_x//4
        sy = num_splits_y//4

        for _ in range(num_splits_y):
            for _ in range(num_splits_x):
                x = round(random.normalvariate(mu=cx, sigma=sx))
                y = round(random.normalvariate(mu=cy, sigma=sy))
                image[x*steps[1]:(x+1)*steps[1], y*steps[0]:(y+1)*steps[0], :] = \
                    self._elements[random.choice(list(normal_element_ids))]
        return image

    def _assign_display_hsb_colors(self, elements):
        """
        RGB colors for element

        Note: Unused
        """
        def get_h_avg(h):
            if h[0] > h[1]:
                return 0
            return h[0] + (h[1] - h[0]) / 2
        self._elements = dict()
        for el in elements:
            self._elements[el["id"]] = \
                hsl2rgb(
                    get_h_avg(el["h"]),
                    el["s"][0] + (el["s"][1] - el["s"][0]) / 2,
                    el["b"][0] + (el["b"][1] - el["b"][0]) / 2
                )
