from abc import ABC, abstractmethod


class ModelProvider(ABC):
    @abstractmethod
    def provide(self, src, tgt):
        pass


class MarianModelProvider(ModelProvider):
    def provide(self, src, tgt):
        return f'Helsinki-NLP/opus-mt-{src.split(".")[-1]}-{tgt.split(".")[-1]}'


class MBart50ModelProvider(ModelProvider):
    def __init__(self, m2m: bool = False):
        self._m2m = m2m

    def provide(self, src, tgt):
        s = 'many' if self._m2m else 'one'
        return f'facebook/mbart-large-50-{s}-to-many-mmt'
