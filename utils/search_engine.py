import time
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict
from himpy.executor import Parser, Evaluator
from himpy.histogram import Histogram
from himpy.utils import E
import ctypes
import platform

if platform.uname()[0] == "Windows":
    lib_name = ".\\mylibrary.dll"
elif platform.uname()[0] == "Linux":
    lib_name = "./library.so"
else:
    lib_name = "./library.dylib"


class BaseSearchEngine(ABC):
    @abstractmethod
    def retrieve(
            self, query: Union[E, Histogram],
            top_n: Union[int, None],
            last_n: Union[int, None],
            threshold: float):
        pass


class DefaultSearchEngine(BaseSearchEngine):
    """Simple search engine based on iterating over entire datasets."""

    def __init__(self, hists: List[Tuple[int, Histogram]], parser: Parser, evaluator: Evaluator):
        self._hists = hists
        self._parser = parser
        self._evaluator = evaluator

    def retrieve(
            self, query: Union[E, Histogram],
            top_n: Union[int, None] = 10,
            last_n: Union[int, None] = None,
            threshold: float = 0.001):
        img_rank = list()
        if hasattr(query, "value") and isinstance(query.value, str):
            """Searching by expression"""
            expression = ["(" + ", ".join(e) + ")" if isinstance(e, tuple) else e for e in self._parser.parse_string(query.value)]
            scores = [(doc_id, self._evaluator.eval(expression, hist).sum()) for doc_id, hist in self._hists]
            img_rank = sorted(
                [(doc_id, score) for doc_id, score in scores if score > threshold],
                key=lambda x: -x[1]
            )
        elif isinstance(query, Histogram):
            """Searching by data histogram"""
            img_rank = sorted(
                [(doc_id, (query * hist).sum()) for doc_id, hist in self._hists],
                key=lambda x: -x[1]
            )
        if isinstance(last_n, int):
            return img_rank[:top_n], img_rank[-last_n:]

        return img_rank[:top_n]


class InvertedIndex(BaseSearchEngine):
    """Search engine based on inverted indexes of histogram elements."""

    def __init__(self, hists: List[Tuple[int, Histogram]], parser: Parser, evaluator: Evaluator):
        self._parser = parser
        self._evaluator = evaluator
        self._storage = dict()
        self._hists = dict()
        for hist_id, hist in hists:
            self._hists[hist_id] = hist
            for index in hist.elements():
                self._storage.setdefault(index, set()).add(hist_id)

    def retrieve(
            self, query: Union[E, Histogram],
            top_n: Union[int, None] = 10,
            last_n: Union[int, None] = None,
            threshold: float = 0.001):
        scores = []
        doc_ids_set = set()
        if hasattr(query, "value") and isinstance(query.value, str):
            """Searching by expression"""
            # expression = self._parser.parse_string(query.value)
            expression = ["(" + ", ".join(e) + ")" if isinstance(e, tuple) else e for e in self._parser.parse_string(query.value)]
            doc_ids_set = self._evaluator.eval_expression(expression, self._storage)
            for doc_id in doc_ids_set:
                scores.append((doc_id, self._evaluator.eval(expression, self._hists[doc_id]).sum()))

        elif isinstance(query, Histogram):
            """Searching by data histogram"""
            indexes_set = query.elements()
            for index in indexes_set:
                if index in self._storage:
                    doc_ids_set.update(self._storage[index])
            for doc_id in doc_ids_set:
                scores.append((doc_id, (query * self._hists[doc_id]).sum()))

        docs_ranked = sorted(
            [(doc_id, score) for doc_id, score in scores if score > threshold],
            key=lambda x: -x[1]
        )

        if isinstance(last_n, int):
            return docs_ranked[:top_n], docs_ranked[-last_n:]
        return docs_ranked[:top_n]


libinvertedindex = ctypes.cdll.LoadLibrary(lib_name)
libinvertedindex.createInvertedIndex.restype = ctypes.c_void_p
libinvertedindex.deleteInvertedIndex.argtypes = [ctypes.c_void_p]
libinvertedindex.addDocument.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
libinvertedindex.retrieveByQuerySingle.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
libinvertedindex.retrieveByQuerySingle.restype = ctypes.c_void_p
libinvertedindex.retrieveByQuery.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
libinvertedindex.retrieveByQuery.restype = ctypes.c_void_p
libinvertedindex.retrieveByHistogramSingle.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
libinvertedindex.retrieveByHistogramSingle.restype = ctypes.c_void_p
libinvertedindex.retrieveByHistogram.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
libinvertedindex.retrieveByHistogram.restype = ctypes.c_void_p

libinvertedindex.addOneDimensionalRules.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libinvertedindex.addMultiDimensionalRules.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libinvertedindex.evalHistogram.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
libinvertedindex.evalHistogram.restype = ctypes.c_void_p
libinvertedindex.evalExpression.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
libinvertedindex.evalExpression.restype = ctypes.c_void_p

libinvertedindex.newMapStringDouble.restype = ctypes.c_void_p
libinvertedindex.insertIntoMapStringDouble.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
libinvertedindex.deleteMapStringDouble.argtypes = [ctypes.c_void_p]

libinvertedindex.newVectorString.restype = ctypes.c_void_p
libinvertedindex.insertIntoVectorString.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libinvertedindex.deleteVectorString.argtypes = [ctypes.c_void_p]

libinvertedindex.getFromVectorIntDouble.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
libinvertedindex.deleteVectorIntDouble.argtypes = [ctypes.c_void_p]

libinvertedindex.getFromVectorStringDouble.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
libinvertedindex.getFromVectorStringDouble.restype = ctypes.c_char_p
libinvertedindex.deleteVectorStringDouble.argtypes = [ctypes.c_void_p]

libinvertedindex.getFirstFromPairVectorIntVectorString.argtypes = [ctypes.c_void_p, ctypes.c_int]
libinvertedindex.getFirstFromPairVectorIntVectorString.restype = ctypes.c_int
libinvertedindex.getSecondFromPairVectorIntVectorString.argtypes = [ctypes.c_void_p, ctypes.c_int]
libinvertedindex.getSecondFromPairVectorIntVectorString.restype = ctypes.c_char_p
libinvertedindex.deletePairVectorIntVectorString.argtypes = [ctypes.c_void_p]


libinvertedindex.newVectorPairStringVectorString.restype = ctypes.c_void_p
libinvertedindex.pushOuterToVectorPairStringVectorString.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libinvertedindex.pushInnerToVectorPairStringVectorString.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libinvertedindex.deleteVectorPairStringVectorString.argtypes = [ctypes.c_void_p]

libinvertedindex.newVectorPairStringVectorInt.restype = ctypes.c_void_p
libinvertedindex.pushOuterToVectorPairStringVectorInt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libinvertedindex.pushInnerToVectorPairStringVectorInt.argtypes = [ctypes.c_void_p, ctypes.c_int]
libinvertedindex.deleteVectorPairStringVectorInt.argtypes = [ctypes.c_void_p]


def encodeVectorString(data: list[str]):
    cpp_expr = libinvertedindex.newVectorString()
    for item in data:
        libinvertedindex.insertIntoVectorString(cpp_expr, item.encode('utf-8'))
    return cpp_expr


def encodeMapStringDouble(data: dict[str, float]):
    cpp_map = libinvertedindex.newMapStringDouble()
    for key, value in data.items():
        libinvertedindex.insertIntoMapStringDouble(cpp_map, key.encode('utf-8'), value)
    return cpp_map


def encodeVectorPairStringVectorString(data: dict[str, set[str]]):
    cpp_vec = libinvertedindex.newVectorPairStringVectorString()
    for key, values in data.items():
        libinvertedindex.pushOuterToVectorPairStringVectorString(cpp_vec, key.encode('utf-8'))
        for value in values:
            libinvertedindex.pushInnerToVectorPairStringVectorString(cpp_vec, value.encode('utf-8'))
    return cpp_vec


def decodeVectorIntDouble(data, size) -> list[tuple[int, float]]:
    result = []
    for i in range(0, size.value):
        first = ctypes.c_int()
        second = ctypes.c_double()
        libinvertedindex.getFromVectorIntDouble(data, i, first, second)
        result.append((first.value, second.value))
    libinvertedindex.deleteVectorIntDouble(data)
    return result


class InvertedIndexCpp(BaseSearchEngine):
    """Search engine based on inverted indexes of histogram elements."""

    def __init__(self, hists: List[Tuple[int, Histogram]], parser: Parser, rules):
        self._parser = parser
        self._index = libinvertedindex.createInvertedIndex()
        if isinstance(rules, list) or isinstance(rules, list):
            self._is_multi = True
            self._is_single = False
            for rule in rules:
                cpp_vec = encodeVectorPairStringVectorString(rule)
                libinvertedindex.addMultiDimensionalRules(self._index, cpp_vec)
        else:
            self._is_multi = False
            self._is_single = True
            cpp_vec = encodeVectorPairStringVectorString(rules)
            libinvertedindex.addOneDimensionalRules(self._index, cpp_vec)
        for hist_id, hist in hists:
            cpp_map = encodeMapStringDouble(hist.to_dict())
            libinvertedindex.addDocument(self._index, hist_id, cpp_map)
            libinvertedindex.deleteMapStringDouble(cpp_map)

    def retrieve(self, query: Union[E, Histogram], top_n: Union[int, None] = 10, last_n: Union[int, None] = None, threshold: float = 0.001):
        size = ctypes.c_int()
        result = []
        if hasattr(query, "value") and isinstance(query.value, str):
            """Searching by expression"""
            # start_time = time.time()
            # expression = ["(" + ", ".join(e) + ")" if isinstance(e, tuple) else e for e in
            #               self._parser.parse_string(query.value)]
            # end_time = time.time()
            # execution_time = (end_time - start_time) * 1000
            # print("Expression build in milliseconds:", execution_time)
            # # expression = ["(" + ", ".join(e) + ")" if isinstance(e, tuple) else e for e in self._parser.parse_string(query.value)]
            # start_time = time.time()
            # cpp_expr = encodeVectorString(expression)
            # end_time = time.time()
            # execution_time = (end_time - start_time) * 1000
            # print("cpp_expr build in milliseconds:", execution_time)
            expression = ["(" + ", ".join(e) + ")" if isinstance(e, tuple) else e for e in self._parser.parse_string(query.value)]
            cpp_expr = encodeVectorString(expression)
            if top_n:
                # start_time = time.time()
                # result = libinvertedindex.retrieveByQuery(self._index, cpp_expr, top_n, False, threshold, size)
                # end_time = time.time()
                # execution_time = (end_time - start_time) * 1000
                # print("retrieveByQuery build in milliseconds:", execution_time)
                result = libinvertedindex.retrieveByQuery(self._index, cpp_expr, top_n, False, threshold, size)
            else:
                result = libinvertedindex.retrieveByQuerySingle(self._index, cpp_expr, last_n, True, threshold, size)
            # start_time = time.time()
            # libinvertedindex.deleteVectorString(cpp_expr)
            # end_time = time.time()
            # execution_time = (end_time - start_time) * 1000
            # print("deleteVectorString build in milliseconds:", execution_time)
            libinvertedindex.deleteVectorString(cpp_expr)
        elif isinstance(query, Histogram):
            """Searching by data histogram"""
            cpp_map = encodeMapStringDouble(query.to_dict())
            if top_n:
                result = libinvertedindex.retrieveByHistogram(self._index, cpp_map, top_n, False, threshold, size)
            else:
                result = libinvertedindex.retrieveByHistogram(self._index, cpp_map, last_n, True, threshold, size)
            libinvertedindex.deleteMapStringDouble(cpp_map)
        # start_time = time.time()
        # data = decodeVectorIntDouble(result, size)
        # end_time = time.time()
        # execution_time = (end_time - start_time) * 1000
        # print("decodeVectorIntDouble build in milliseconds:", execution_time)
        # return data
        return decodeVectorIntDouble(result, size)
    
    def __del__(self):
        libinvertedindex.deleteInvertedIndex(self._index)



class SearchEngine:
    def __init__(self, hists: List[Tuple[int, Histogram]], parser: Parser, evaluator: Evaluator, use_index=True):
        if use_index:
            self._search_engine = InvertedIndex(hists, parser, evaluator)
        else:
            self._search_engine = DefaultSearchEngine(hists, parser, evaluator)

    def retrieve(
            self, query: Union[E, Histogram],
            top_n: Union[int, None] = 10,
            last_n: Union[int, None] = None,
            threshold: float = 0.001):
        return self._search_engine.retrieve(query, top_n, last_n, threshold)

