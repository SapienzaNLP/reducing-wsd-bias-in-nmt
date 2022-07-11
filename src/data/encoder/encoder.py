from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional

from data.encoder.model_provider import ModelProvider, MarianModelProvider
from data.entities import WSDMTParallelItem, ContextualizedAnnotatedSpan

if TYPE_CHECKING:
    from data.wsdmt_dataset import WSDMTDataset


class SentenceBatchEncoder(ABC):
    _dataset = None

    def __init__(self, model_provider: ModelProvider = None):
        self.provider = model_provider or MarianModelProvider()

    def link(self, dataset):
        if self._dataset is not None:
            return
        self._dataset = dataset

    @property
    def model_name(self):
        src, tgt = self.dataset.langs
        return self.provider.provide(src, tgt)

    @property
    def dataset(self) -> 'WSDMTDataset':
        assert self._dataset is not None, "invoke encoder.link(dataset) at least once before using the encoder!"
        return self._dataset

    def fit(self, dataset: 'WSDMTDataset'):
        if self._dataset is None:
            self._dataset = dataset
        return self._fit(dataset)

    @abstractmethod
    def _fit(self, dataset: 'WSDMTDataset'):
        pass

    @property
    def langs(self):
        return self.dataset.langs

    @property
    def tokenizer(self):
        return self.dataset.tokenizer

    def use_span(self, span: ContextualizedAnnotatedSpan) -> bool:
        return True

    @abstractmethod
    def encode(self, items: List[WSDMTParallelItem]):
        pass

    @abstractmethod
    def encode_sentences(self, src_sentences: List[str], tgt_sentences: Optional[List[str]] = None):
        pass

    @abstractmethod
    def decode_ids(self, ids):
        pass
