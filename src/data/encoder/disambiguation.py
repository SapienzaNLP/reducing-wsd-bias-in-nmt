from typing import List, Optional, Iterable, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from data.encoder.basic import BasicSentenceBatchEncoder
from data.encoder.model_provider import ModelProvider
from data.encoder.sense import SenseSentenceBatchEncoder
from data.entities import AnnotatedSpan, WSDMTParallelItem, WSDMTItem, ContextualizedAnnotatedSpan


class EncoderDisambiguationSentenceBatchEncoder(SenseSentenceBatchEncoder):
    def __init__(self, model_provider: ModelProvider = None):
        super().__init__(model_provider)
        self.label_vocab['NOSENSE'] = 0
        self.inverse_label_vocab[0] = 'NOSENSE'

    def use_span(self, span: ContextualizedAnnotatedSpan) -> bool:
        return span.sense is not None  # and span.identified_as_sense and span.is_polysemous and span.sense[-1] == 'n'

    def encode(self, items: List[WSDMTParallelItem]):
        batch = BasicSentenceBatchEncoder.encode(self, items)

        src_text_ids = []
        synsets_ids = []

        for item in items:
            encodings, synsets = self.encode_item(item.src_item)
            src_text_ids.append(torch.tensor(encodings))
            synsets_ids.append(torch.tensor(synsets))

        src_ids_tensor = pad_sequence(src_text_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        synsets_ids_tensor = pad_sequence(synsets_ids, batch_first=True, padding_value=self.label_vocab['NOSENSE'])

        batch['input_ids'] = src_ids_tensor
        batch['attention_mask'] = (src_ids_tensor != self.tokenizer.pad_token_id).long()
        batch['input_senses'] = synsets_ids_tensor
        batch['sense_mask'] = (synsets_ids_tensor != self.label_vocab['NOSENSE']).long()

        return batch

    def choose_span_label(self, span: ContextualizedAnnotatedSpan) -> Optional[str]:
        return span.sense

    def encode_item(self, item: WSDMTItem):
        encodings = []
        synsets = []

        max_len = self.tokenizer.model_max_length - 1

        for span in item.spans:
            token = span.text
            subword_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
            if len(encodings) + len(subword_ids) > max_len:
                break

            encodings.extend(subword_ids)

            label = self.choose_span_label(span) if self.use_span(span) else 'NOSENSE'
            sense_index = self.label_vocab[label]
            synsets.extend([sense_index] * len(subword_ids))

        encodings.append(self.tokenizer.eos_token_id)
        synsets.append(self.label_vocab['NOSENSE'])

        return encodings, synsets


class DecoderDisambiguationSentenceBatchEncoder(EncoderDisambiguationSentenceBatchEncoder):
    # _CLUSTERING = load_synset_clustering()

    def use_span(self, span: AnnotatedSpan, item: Optional[WSDMTItem] = None) -> bool:
        return span.sense is not None  # and span.identified_as_sense and span.is_polysemous and span.sense[-1] == 'n'

    def _item_spans(self, item: WSDMTParallelItem) -> Iterable[ContextualizedAnnotatedSpan]:
        return item.tgt_item.spans

    def encode(self, items: List[WSDMTParallelItem]):
        batch = BasicSentenceBatchEncoder.encode(self, items)

        tgt_text_ids = []
        synsets_ids = []

        for item in items:
            encodings, synsets = self.encode_item(item.tgt_item)
            tgt_text_ids.append(torch.tensor(encodings))
            synsets_ids.append(torch.tensor(synsets))

        tgt_ids_tensor = pad_sequence(tgt_text_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        synsets_ids_tensor = pad_sequence(synsets_ids, batch_first=True, padding_value=self.label_vocab['NOSENSE'])

        batch['labels'] = tgt_ids_tensor
        # batch['attention_mask'] = (tgt_ids_tensor != self.tokenizer.pad_token_id).long()
        batch['input_senses'] = synsets_ids_tensor
        batch['sense_mask'] = (synsets_ids_tensor != self.label_vocab['NOSENSE']).long()

        return batch

    def choose_span_label(self, span: AnnotatedSpan) -> Optional[str]:
        return span.sense

    def encode_item(self, item: WSDMTItem):
        encodings = []
        synsets = []

        max_len = self.tokenizer.model_max_length - 1

        with self.tokenizer.as_target_tokenizer():
            for span in item.spans:
                token = span.text
                subword_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))

                if len(encodings) + len(subword_ids) > max_len:
                    break

                encodings.extend(subword_ids)

                label = self.choose_span_label(span) if self.use_span(span, item) else 'NOSENSE'
                sense_index = self.label_vocab[label]
                synsets.extend([sense_index] * len(subword_ids))

        encodings.append(self.tokenizer.eos_token_id)
        synsets.append(self.label_vocab['NOSENSE'])

        return encodings, synsets


class EncDecDisambiguationSentenceBatchEncoder(EncoderDisambiguationSentenceBatchEncoder):
    def __init__(self, model_provider: ModelProvider = None):
        super().__init__(model_provider)
        self._encoder_side = EncoderDisambiguationSentenceBatchEncoder(model_provider)
        self._decoder_side = DecoderDisambiguationSentenceBatchEncoder(model_provider)

    def link(self, dataset):
        super().link(dataset)
        self._encoder_side.link(dataset)
        self._decoder_side.link(dataset)

    def update_label_vocab(self, dataset):
        self._encoder_side.update_label_vocab(dataset)
        self._decoder_side.senses_mapping = self._encoder_side.senses_mapping
        self.senses_mapping = self._encoder_side.senses_mapping

    def _item_spans(self, item: WSDMTParallelItem) -> Iterable[ContextualizedAnnotatedSpan]:
        return *item.src_item.spans, *item.tgt_item.spans

    def encode(self, items: List[WSDMTParallelItem]):
        enc_batch = self._encoder_side.encode(items)
        dec_batch = self._decoder_side.encode(items)

        enc_batch['labels'] = dec_batch['labels']
        enc_batch['labels_input_senses'] = dec_batch['input_senses']
        enc_batch['labels_sense_mask'] = dec_batch['sense_mask']

        return enc_batch
