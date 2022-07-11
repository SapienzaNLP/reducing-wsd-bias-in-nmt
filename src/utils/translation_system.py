from dataclasses import dataclass
from typing import List, Union, Dict

from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from more_itertools import ichunked

import torch
import torch.nn.functional as F

from data.encoder import SentenceBatchEncoder
from data.wsdmt_dataset import WSDMTDataset
from utils.evaluation import evaluate, EvaluationResult
from model.base import BaseMTModel

import wandb


@dataclass
class TranslationCorpus:
    name: str
    sentences: List[str]
    references: List[str]
    predictions: List[str] = None
    metrics: EvaluationResult = None

    def translate_with(self, system: 'TranslationSystem', batch_size=64) -> 'TranslationCorpus':
        desc = f"Computing translations for {self.name}"
        self.predictions, self.metrics = system.evaluate_corpus(self.sentences, self.references, desc, batch_size)
        return self

    def wandb_table(self) -> wandb.Table:
        assert self.predictions is not None, "Cannot invoke .wandb_table() before having computed the translations"

        references_joined = map(lambda e: e if isinstance(e, str) else '; '.join(e), self.references)
        rows = list(zip(self.sentences, references_joined, self.predictions))
        table = wandb.Table(columns=['Source', 'Target', 'Prediction'], rows=rows)
        return table

    def __str__(self):
        if self.metrics:
            info_str = self.metrics.metrics_str()
        else:
            info_str = 'translated=no'
        return f"Corpus<{self.name}, size={len(self.sentences)}>[{info_str}]"


class TranslationSystem:
    def __init__(self,
                 model: BaseMTModel,
                 encoder: SentenceBatchEncoder,
                 num_beams: int = 1,
                 ):
        self.model = model
        self.encdec = model.model

        self.tokenizer = encoder.tokenizer
        self.decode = encoder.decode_ids
        self.encode = encoder.encode_sentences

        self.src_lang = encoder.dataset.src_lang
        self.tgt_lang = encoder.dataset.tgt_lang

        self._bad_words = [[_id] for _id in self.tokenizer.additional_special_tokens_ids] or None
        self.num_beams = num_beams

    @property
    def device(self):
        return self.model.device

    def batch_decode(self, ids: torch.Tensor):
        return [self.decode(_ids) for _ids in ids]

    def _lang_code_for(self, lang):
        return next(k for k in self.tokenizer.lang_code_to_id if k[:2] == lang)

    def translate_batch(self, batch, beams=None, attentions=False):
        batch.pop('sids', None)
        batch = move_data_to_device(batch, self.device)
        input_ids = batch['input_ids']

        assert len(input_ids.shape) == 2
        beams = beams or self.num_beams

        with torch.no_grad():
            bos_token_id = None
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                bos_token_id = self.tokenizer.lang_code_to_id[self._lang_code_for(self.tgt_lang)]

            with self.model.fix_position_ids(batch.get('position_ids', None)):

                out = self.encdec.generate(input_ids=input_ids,
                                           num_beams=beams,
                                           bad_words_ids=self._bad_words,
                                           output_attentions=attentions,
                                           forced_bos_token_id=bos_token_id,
                                           early_stopping=True)

            result = self.batch_decode(out[:, 1:])
            return result

    def translate(self, sentences: Union[str, List[str]], beams=None) -> Union[str, List[str]]:
        is_single_sentence = isinstance(sentences, str)
        if is_single_sentence:
            sentences = [sentences]

        batch = self.encode(sentences)
        result = self.translate_batch(batch, beams)
        return result[0] if is_single_sentence else result

    def score(self, sentences: Union[str, List[str]], translations: Union[str, List[str]]):

        is_single_sentence = isinstance(sentences, str)
        if is_single_sentence:
            assert isinstance(translations, str)
            sentences = [sentences]
            translations = [translations]

        with torch.no_grad():
            batch = self.encode(sentences, translations)
            lengths = (batch['labels'] != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
            output = self.encdec.forward(**batch.to(self.device))
            batch_logits = output.logits

            scores = []
            # xent = F.cross_entropy()
            for labels, logits, length in zip(batch['labels'], batch_logits, lengths):
                score = F.cross_entropy(logits[:length], labels[:length], ignore_index=self.tokenizer.pad_token_id)
                scores.append(score.item())

        return scores[0] if is_single_sentence else scores

    def predict(self, source_sentence, target_sentence):
        batch = self.encode([source_sentence], [target_sentence])
        with torch.no_grad():
            output = self.encdec(**batch.to(self.device), output_attentions=True, return_dict=True)
            return output, batch

    def translate_corpus(self,
                         source_sentences: List[str],
                         desc=None,
                         batch_size=64):
        iterator = iter(source_sentences)
        if desc is not None:
            iterator = tqdm(iterator, desc=desc, leave=False)
        iterator = ichunked(iterator, batch_size)

        return (prediction
                for sentence_batch in iterator
                for prediction in self.translate(list(sentence_batch)))

    def translate_dataset(self, dataset: WSDMTDataset, desc=None, batch_size=64):
        sources = list(item.src_item.sentence for item in dataset)
        return self.translate_corpus(sources, desc=desc, batch_size=batch_size)

    def evaluate_corpus(self,
                        source_sentences: List[str],
                        target_sentences: List[str],
                        desc=None,
                        batch_size=64):
        predictions = list(self.translate_corpus(source_sentences, desc, batch_size))
        scores = evaluate(predictions, target_sentences, self.tgt_lang)
        return predictions, scores

    def evaluate_dataset(self, dataset: WSDMTDataset, batch_size=64):
        sources, references = zip(*((item.src_item.sentence, item.tgt_item.sentence) for item in dataset))
        corpus = TranslationCorpus(dataset.name, sources, references)
        return corpus.translate_with(self, batch_size)

    def evaluate_loader(self, loader: DataLoader, name):
        source_sentences = []
        target_sentences = []
        predictions = []

        for batch in tqdm(loader, desc=f"Evaluating {name}", leave=False):
            input_ids = batch['input_ids']
            source_sentences.extend(self.batch_decode(input_ids))
            target_sentences.extend(self.batch_decode(batch['labels']))
            predictions.extend(self.translate_batch(batch))

        metrics = evaluate(predictions, target_sentences, self.tgt_lang)
        corpus = TranslationCorpus(f"{name}_loader", source_sentences, target_sentences, predictions, metrics)
        return corpus
