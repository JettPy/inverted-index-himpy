from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from himpy.executor import Parser, Evaluator
from himpy.histogram import Histogram
from himpy.utils import E


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
            expression = self._parser.parse_string(query.value)
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
            expression = self._parser.parse_string(query.value)
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
