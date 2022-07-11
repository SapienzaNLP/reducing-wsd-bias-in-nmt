from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from data.wsdmt_dataset import WSDMTDataset


@dataclass
class AnnotatedSpan:
    idx: int
    text: str
    sense: Optional[str]
    identified_as_sense: bool
    is_polysemous: bool
    lemma: str


@dataclass
class ContextualizedAnnotatedSpan(AnnotatedSpan):
    item: 'WSDMTParallelItem'
    lang: str


@dataclass
class WSDMTParallelItem:
    _dataset: 'WSDMTDataset'
    index: int

    @property
    def item(self):
        return self._dataset.dataset[self.index]

    @property
    def items(self) -> Dict[str, 'WSDMTItem']:
        return {lang: self[lang] for lang in self._dataset.langs}

    @property
    def src_item(self) -> 'WSDMTItem':
        return self[self._dataset.src]

    @property
    def tgt_item(self) -> 'WSDMTItem':
        return self[self._dataset.tgt]

    @property
    def sid(self) -> str:
        return self.item['sid']

    @property
    def span_dict(self) -> Dict[str, List[ContextualizedAnnotatedSpan]]:
        return {lang: list(self[lang].spans) for lang in self._dataset.langs}

    @property
    def common_senses(self):
        return self.src_item.senses.intersection(self.tgt_item.senses)

    def __getitem__(self, item) -> 'WSDMTItem':
        assert item in self._dataset.langs
        return WSDMTItem(self, item)

    def __repr__(self):
        return f"ParallelItem<index={self.index}, sid={self.sid}>"


@dataclass
class WSDMTItem:
    _parallel_item: WSDMTParallelItem
    lang: str

    @property
    def spans(self) -> Iterator[ContextualizedAnnotatedSpan]:
        item = self._parallel_item.item[self.lang]

        keys = ['tokens', 'sense', 'identified_as_sense', 'is_polysemous', 'lemmas']
        iterator = map(lambda k: item[k], keys)

        for i, values in enumerate(zip(*iterator)):
            span = ContextualizedAnnotatedSpan(i, *values, lang=self.lang, item=self._parallel_item)
            yield span

    @property
    def sentence(self) -> str:
        return self.parallel_item.item[self.lang]['sentence']

    @property
    def parallel_item(self):
        return self._parallel_item

    @property
    def senses(self):
        return {span.sense for span in self.spans if span.sense is not None}

    def __repr__(self):
        item = self._parallel_item
        return f"Item<index={item.index}, sid={item.sid}, lang={self.lang}>"
