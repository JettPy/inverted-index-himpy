from .histogram import Histogram, HElementSet


class E:

    def __init__(self, *expression):
        self.value = "(" + ",".join(expression) + ")"

    def Union(self, other):
        return self._compose(other.value, "+")

    def Intersection(self, other):
        return self._compose(other.value, "*")

    def Sub(self, other):
        return self._compose(other.value, "/")

    def Or(self, other):
        return self._compose(other.value, "|")

    def And(self, other):
        return self._compose(other.value, "&")

    def Xor(self, other):
        return self._compose(other.value, "#|")

    def Xsub(self, other):
        return self._compose(other.value, "#/")

    def _compose(self, other, op):
        if isinstance(other, E):
            return E(self.value + op + other.value)
        elif isinstance(other, str):
            return E(self.value + op + other)

    def __add__(self, other):
        return self.Union(other)

    def __mul__(self, other):
        return self.Intersection(other)

    def __sub__(self, other):
        return self.Sub(other)

    def __and__(self, other):
        return self.And(other)

    def __or__(self, other):
        return self.Or(other)

    def __xor__(self, other):
        return self.Xor(other)

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


def convert_hist_to_all_values(U, H, to_sort=False):
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
