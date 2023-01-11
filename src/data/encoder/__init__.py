from typing import Type

from data.encoder.encoder import SentenceBatchEncoder
from data.encoder.basic import BasicSentenceBatchEncoder
from data.encoder.sense import SenseSentenceBatchEncoder
from data.encoder.random import RandomSenseSentenceBatchEncoder
from data.encoder.model_provider import MarianModelProvider, MBart50ModelProvider, ModelProvider

_ENCODERS = dict(
    basic=BasicSentenceBatchEncoder,
    sense=SenseSentenceBatchEncoder,
    random=RandomSenseSentenceBatchEncoder,
)


_PROVIDERS = {
    'opus': MarianModelProvider,
    'mbart50': MBart50ModelProvider,
    'mbart50-mtm': lambda: MBart50ModelProvider(m2m=True),
}


ENCODERS = frozenset(_ENCODERS.keys())
PROVIDERS = frozenset(_PROVIDERS.keys())


def get_encoder(encoder_name: str) -> Type[SentenceBatchEncoder]:
    return _ENCODERS.get(encoder_name)


def get_provider(provider_name: str) -> Type[ModelProvider]:
    return _PROVIDERS.get(provider_name)
