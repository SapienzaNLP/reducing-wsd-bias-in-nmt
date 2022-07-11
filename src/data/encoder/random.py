from typing import Optional

from tqdm.auto import tqdm

from data.encoder import SenseSentenceBatchEncoder
from data.entities import ContextualizedAnnotatedSpan
import numpy as np


class RandomSenseSentenceBatchEncoder(SenseSentenceBatchEncoder):
    def __init__(self):
        super().__init__()
        self.lemma2synsets = {}
        self._updating_vocab = False

    def update_label_vocab(self, dataset):
        vocab = self.label_vocab
        inverse = self.inverse_label_vocab
        self._updating_vocab = True

        for item in tqdm(dataset, desc=f"Updating sense vocabulary with {dataset.name}"):
            spans = self._item_spans(item)
            for span in spans:
                if not self.use_span(span):
                    continue

                label = self.choose_span_label(span)
                self.lemma2synsets.setdefault(span.lemma, set()).add(label)

                if label not in vocab:
                    idx = len(vocab)
                    vocab[label] = idx
                    inverse[idx] = label

        self._updating_vocab = False
        self.lemma2synsets = {
            lemma: list(synsets)
            for lemma, synsets in self.lemma2synsets.items()
        }

    def choose_span_label(self, span: ContextualizedAnnotatedSpan) -> Optional[str]:
        if self._updating_vocab:
            return super().choose_span_label(span)

        if not self.use_span(span):
            return None

        synsets = self.lemma2synsets[span.lemma]
        return np.random.choice(synsets)
