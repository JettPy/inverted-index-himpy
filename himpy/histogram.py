"""
Histogram Data Structures
"""
from typing import Tuple, Union, List, Any, Set, Dict, Callable


class Element:
    pass


class HElement:
    """
    Low Level Histogram Element

    This element must corresponds to one of elements from the universal set.

    Note: The universal set is one from which data is made up. Think of it as
    a dictionary of terms.

    Parameters
    ----------
    key         an element id
    value       a value of the element
    properties  additional parameters that can be used in evaluation phase,
    e.g. mean, var

    """

    def __init__(self, key: Union[str, Tuple[str, ...]], value: float, properties: Union[List[Any], None] = None):
        self._key = key
        self._value = value
        self._properties = properties or list()

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other_h_element: 'HElement'):
        return isinstance(other_h_element, HElement) and self._key == other_h_element.key

    def __ne__(self, other_h_element: 'HElement'):
        return isinstance(other_h_element, HElement) and self._key != other_h_element.key


class HElementCollection:
    """Base class for a histogram of elements (HE)"""

    def add(self, h_element: HElement):
        """Add the new element to the histogram"""
        raise NotImplementedError

    def discard(self, h_element: HElement):
        """Discard the element from the histogram"""
        raise NotImplementedError


class HElementSet(HElementCollection):
    """
    Histogram of elements implemented as a set

    Parameters
    ----------
    _h_element_set  set of elements of type HElement
    """

    def __init__(self, h_element_set: Union[Set[HElement], None] = None):
        super(HElementSet, self).__init__()
        self._h_element_set = h_element_set or set()

    def add(self, h_element: HElement):
        if isinstance(h_element, HElement):
            self._h_element_set.add(h_element)
        else:
            raise Exception("Argument is not HElement")

    def discard(self, h_element: HElement):
        if isinstance(h_element, HElement):
            self._h_element_set.discard(h_element)
        else:
            raise Exception("Argument is not HElement")

    def union(self, other_h_element_set: 'HElementSet'):
        if isinstance(other_h_element_set, HElementSet):
            return self._h_element_set.union(other_h_element_set)
        raise Exception("Argument is not HElement")

    def intersection(self, other_h_element_set: 'HElementSet'):
        if isinstance(other_h_element_set, HElementSet):
            return self._h_element_set.intersection(other_h_element_set)
        raise Exception("Argument is not HElement")

    def difference(self, other_h_element_set: 'HElementSet'):
        if isinstance(other_h_element_set, HElementSet):
            return self._h_element_set.difference(other_h_element_set)
        raise Exception("Argument is not HElement")

    def sum(self) -> float:
        """Sum of all elements of histogram"""
        return sum(h_element.value for h_element in self._h_element_set)

    def prod(self) -> float:
        """Product of all elements of histogram"""
        result = 1
        for h_element in self._h_element_set:
            result = result * h_element.value
        return result

    def elements(self) -> List[Union[str, Tuple[str, ...]]]:
        """List of ids of non-zero histogram elements"""
        if hasattr(self, "_h_element_set") and isinstance(self._h_element_set, set):
            return [h_element.key for h_element in self._h_element_set]
        raise Exception("There are no elements.")

    def values(self) -> List[float]:
        """List of values of non-zero histogram elements"""
        if hasattr(self, "_h_element_set") and isinstance(self._h_element_set, set):
            return [h_element.value for h_element in self._h_element_set]
        raise Exception("There are no elements.")

    def to_dict(self) -> Dict[Union[str, Tuple[str, ...]], float]:
        """Dictionary of non-zero histogram elements: {id:value}"""
        if hasattr(self, "_h_element_set") and isinstance(self._h_element_set, set):
            return {h_element.key: h_element.value for h_element in self._h_element_set}
        raise Exception("There are no elements.")

    def __contains__(self, item: Union[HElement, str, Tuple[str, ...]]):
        if isinstance(item, HElement):
            return item in self._h_element_set
        elif isinstance(item, str) or isinstance(item, tuple):
            return HElement(item, 0) in self._h_element_set
        return False

    def __len__(self):
        return len(self._h_element_set)

    def __iter__(self):
        return self._h_element_set.__iter__()


class Histogram:
    """Data Histogram"""

    def __init__(
            self, data: Any, normalized=True, size: Union[float, None] = None,
            transform_func: Callable[[Any], Dict[Union[str, Tuple[str, ...]], HElement]] = None):

        if not data:
            self._histogram_elements = dict()
            return

        if transform_func:
            self._histogram_elements = transform_func(data)
        else:
            self._histogram_elements = Histogram.transform(data)
        self._size = size or sum(h_element.value for h_element in self._histogram_elements.values())
        self._normalized = False
        if normalized:
            self._normalize()
            self._normalized = True

    def sum(self) -> float:
        if not hasattr(self, "_histogram_elements") or not isinstance(self._histogram_elements, dict):
            raise Exception("There are no elements.")
        return sum(self._histogram_elements[key].value for key in self._histogram_elements)

    # TODO: count histogram elements
    # TODO: count elements

    def elements(self) -> List[Union[str, Tuple[str, ...]]]:
        if hasattr(self, "_histogram_elements") and isinstance(self._histogram_elements, dict):
            return list(self._histogram_elements.keys())
        raise Exception("There are no elements.")

    def hist_elements(self) -> Dict[Union[str, Tuple[str, ...]], HElement]:
        if hasattr(self, "_histogram_elements") and isinstance(self._histogram_elements, dict):
            return self._histogram_elements
        raise Exception("There are no elements.")

    def add(self, element):

        if isinstance(element, HElement):
            if element.key not in self._histogram_elements:
                self._histogram_elements[element.key] = HElement(element.key, 0)
            self._histogram_elements[element.key].value += element.value
            self._size += element.value

    def to_dict(self) -> Dict[Union[str, Tuple[str, ...]], float]:
        """Dictionary of non-zero histogram elements: {id:value}"""
        if hasattr(self, "_histogram_elements") and isinstance(self._histogram_elements, dict):
            return {key: h_element.value for key, h_element in self._histogram_elements.items()}
        raise Exception("There are no elements.")

    def normalize(self, size: Union[float, None] = None):
        if size:
            self._size = size
        self._normalize()

    def _normalize(self):
        """Normalize data histogram to 1"""
        for key in self._histogram_elements:
            self._histogram_elements[key].value = float(self._histogram_elements[key].value) / self._size
        self._normalized = True

    def __call__(self, element, composition=None):
        """
        Create a histogram of elements

        Parameters
        ----------
        element         a low- or high-level element
        composition     used for a high-level element to define a set of low-level elements

        Returns
        -------
        histogram of elements (HE) -> HElementSet

        """
        if element in self:
            return HElementSet(h_element_set={self[element]})
        elif element and composition and element in composition and isinstance(composition[element], set):
            return HElementSet(h_element_set={self[el] for el in composition[element] if el in self})
        return HElementSet()

    def __setitem__(self, key, value):
        self._histogram_elements[key] = value

    def __getitem__(self, item):
        return self._histogram_elements[item]

    def __contains__(self, item):
        return item in self._histogram_elements

    def __len__(self):
        return hasattr(self, "_histogram_elements") and len(self._histogram_elements)

    def __add__(self, other):
        return hist_operations["+"].compute(self, other)

    def __mul__(self, other):
        return hist_operations["*"].compute(self, other)

    def __iter__(self):
        return self._histogram_elements.items().__iter__()

    @staticmethod
    def transform(data):
        """
        Convert data to histogram

        Parameters
        ----------
        data            a data composed from elements of the universal set

        Returns
        -------
        dictionary      {element id : value}

        """
        histogram = dict()
        if isinstance(data, list):
            for el in data:
                if el not in histogram:
                    histogram[el] = HElement(el, 0.0)
                histogram[el].value += 1.0
        return histogram


class Histogram1D(Histogram):
    """Data Histogram with 1D positioning"""

    def __call__(self, element, composition=None):
        """
        Create a histogram of elements

        Parameters
        ----------
        element         a low- or high-level element
        composition     used for a high-level element to define a set of low-level elements

        Returns
        -------
        histogram of elements (HE) -> HElementSet

        """

        # element
        if element.find(", ") != -1 or element[0] == "(" and element[-1] == ")":
            element = tuple(element.replace("(", "").replace(")", "").split(", "))
        element_ndim = len(element) if isinstance(element, tuple) else 1

        Es = dict()
        has_compound = False
        if element_ndim == 1:
            Es = {element}
            if composition is not None and element in composition:
                Es = composition[element]
                has_compound = True
        elif element_ndim > 1:
            for i in range(element_ndim):
                Es[i] = {element[i]}
                if composition is not None and i in composition and element[i] in composition[i]:
                    Es[i] = composition[i][element[i]]
                    has_compound = True

        if not has_compound and element in self:
            return HElementSet(h_element_set={self[element]})
        else:
            condition = None
            if element_ndim == 1:
                condition = lambda x: x in Es or "any" in Es
            elif element_ndim > 1:
                condition = lambda x: all([x.split(', ')[i] in Es[i] or "any" in Es[i] for i in range(element_ndim)])
            return HElementSet(h_element_set=set(el for el in self._histogram_elements.values() if condition(el.key)))


class Histogram2D(Histogram):
    """Data Histogram with 2D positioning"""
    pass


class Histogram3D(Histogram):
    """Data Histogram with 3D positioning"""
    pass


class OperationBase:
    """Base class for operations"""

    sign = None
    description = None

    def compute(self, arg1, arg2):
        raise NotImplementedError

    def __call__(self, arg1, arg2):
        return self.compute(arg1, arg2)


"""
Operations over Histogram
"""


class HistUnion(OperationBase):

    sign = "+"
    description = ""

    def compute(self, arg1, arg2):
        opn1, opn2 = (arg2, arg1) if len(arg1) > len(arg2) else (arg1, arg2)
        hist = Histogram(data=None)
        for key, value in opn2:
            hist[key] = HElement(key, value.value)
        for key, value in opn1:
            if key not in hist:
                hist[key] = HElement(key, 0)
            hist[key].value += value.value
        return hist


class HistIntersection:

    sign = "*"
    description = ""

    def compute(self, arg1, arg2):
        opn1, opn2 = (arg2, arg1) if len(arg1) > len(arg2) else (arg1, arg2)
        hist = Histogram(data=None)
        for key, value in opn1:
            if key in opn2:
                hist[key] = HElement(key, min(value.value, opn2[key].value))
        return hist


hist_operations = {
    HistUnion.sign: HistUnion(),
    HistIntersection.sign: HistIntersection()
}


"""
Operation over HElementSet
"""


class SetUnion(OperationBase):

    sign = "+"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet(arg1.union(arg2))


class SetIntersection(OperationBase):

    sign = "*"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet(arg1.intersection(arg2))


class SetSubtraction(OperationBase):

    sign = "/"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet(arg1.difference(arg2))


class SetXSubtraction(OperationBase):

    sign = "#/"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet() if arg2.sum() > 0 else arg1


class SetOr(SetUnion):
    sign = "|"


class SetXOr(OperationBase):

    sign = "#|"
    description = ""

    def compute(self, arg1, arg2):
        return arg1 if arg1.sum() > arg2.sum() else arg2


class SetAnd(OperationBase):

    sign = "&"
    description = ""

    def compute(self, arg1, arg2):
        return arg2 if arg1.sum() > arg2.sum() else arg1


"""
Operation over HElementDict
"""


class DictUnion(OperationBase):

    def compute(self, arg1, arg2):
        opn1, opn2 = (arg2, arg1) if len(arg1) > len(arg2) else (arg1, arg2)
        for k, v in opn1:
            opn2.add(v)
        return opn2


operations = {
    SetUnion.sign: SetUnion(),
    SetIntersection.sign: SetIntersection(),
    SetAnd.sign: SetAnd(),
    SetOr.sign: SetOr(),
    SetXOr.sign: SetXOr(),
    SetSubtraction.sign: SetSubtraction(),
    SetXSubtraction.sign: SetXSubtraction()
}

######################################


class ExpressionUnion(OperationBase):

    sign = "+"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        return arg_d1.union(arg_d2), arg_k1.union(arg_k2)


class ExpressionIntersection(OperationBase):

    sign = "*"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        new_keys = arg_k1.intersection(arg_k2)
        if len(new_keys) > 0:
            return arg_d1.intersection(arg_d2), new_keys
        else:
            return set(), set()


class ExpressionSubtraction(OperationBase):

    sign = "/"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        return arg_d1, arg_k1.difference(arg_k2)


class ExpressionAnd(OperationBase):

    sign = "&"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        return arg_d1.intersection(arg_d2), arg_k1.union(arg_k2)


class ExpressionOr(OperationBase):

    sign = "|"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        return arg_d1.union(arg_d2), arg_k1.union(arg_k2)

    
class ExpressionXOr(OperationBase):

    sign = "#|"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        return arg_d1.symmetric_difference(arg_d2), arg_k1.union(arg_k2)


class ExpressionXSubtraction(OperationBase):

    sign = "#/"
    description = ""

    def compute(self, arg1, arg2):
        arg_d1, arg_k1 = arg1
        arg_d2, arg_k2 = arg2
        return arg_d1.difference(arg_d2), arg_k1.difference(arg_k2)

    
expressionOperations = {
    ExpressionUnion.sign: ExpressionUnion(),
    ExpressionIntersection.sign: ExpressionIntersection(),
    ExpressionSubtraction.sign: ExpressionSubtraction(),
    ExpressionAnd.sign: ExpressionAnd(),
    ExpressionOr.sign: ExpressionOr(),
    ExpressionXOr.sign: ExpressionXOr(),
    ExpressionXSubtraction.sign: ExpressionXSubtraction()
}
    