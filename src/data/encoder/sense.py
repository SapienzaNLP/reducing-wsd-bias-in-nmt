from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from data.encoder.basic import BasicSentenceBatchEncoder
from data.encoder.model_provider import ModelProvider
from data.entities import AnnotatedSpan, WSDMTParallelItem, WSDMTItem, ContextualizedAnnotatedSpan

if TYPE_CHECKING:
    from data.wsdmt_dataset import WSDMTDataset


class SenseSentenceBatchEncoder(BasicSentenceBatchEncoder):
    output_prefix = ''

    def __init__(self, model_provider: ModelProvider = None):
        super().__init__(model_provider)
        self.label_vocab: Dict[str, int] = {}
        self.inverse_label_vocab: Dict[int, str] = {}

    def _fit(self, dataset: 'WSDMTDataset'):
        self.update_label_vocab(dataset)

    @property
    def senses_mapping(self):
        return self.label_vocab

    @senses_mapping.setter
    def senses_mapping(self, value):
        self.label_vocab = value
        self.inverse_label_vocab = {v: k for k, v in value.items()}

    @property
    def num_senses(self):
        return len(self.label_vocab)

    def _item_spans(self, item: WSDMTParallelItem) -> Iterable[ContextualizedAnnotatedSpan]:
        return item.src_item.spans

    def update_label_vocab(self, dataset):
        vocab = self.label_vocab
        inverse = self.inverse_label_vocab

        for item in tqdm(dataset, desc=f"Updating sense vocabulary with {dataset.name}"):
            spans = self._item_spans(item)
            for span in spans:
                if not self.use_span(span):
                    continue

                label = self.choose_span_label(span)
                if label not in vocab:
                    idx = len(vocab)
                    vocab[label] = idx
                    inverse[idx] = label

    def use_span(self, span: ContextualizedAnnotatedSpan) -> bool:
        return span.sense is not None  # and span.identified_as_sense and span.is_polysemous and span.sense[-1] == 'n'

    def encode(self, items: List[WSDMTParallelItem]):
        batch = super().encode(items)

        src_text_ids = []
        position_ids = []

        for item in items:
            encodings, positions = self.encode_item(item.src_item)
            src_text_ids.append(torch.tensor(encodings))
            position_ids.append(torch.tensor(positions))

        src_ids_tensor = pad_sequence(src_text_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        position_ids_tensor = pad_sequence(position_ids, batch_first=True, padding_value=0)

        batch[self.output_prefix + 'input_ids'] = src_ids_tensor
        batch[self.output_prefix + 'attention_mask'] = (src_ids_tensor != self.tokenizer.pad_token_id).long()
        batch[self.output_prefix + 'position_ids'] = position_ids_tensor

        return batch

    def decode_ids(self, ids):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)

        assert len(ids.shape) == 1
        offset = len(self.tokenizer)
        synsets = [(idx.item(), ids[idx].item()) for idx in torch.nonzero(ids >= offset).squeeze(-1)]

        # replace synset_ids with UNK to avoid sentencepiece crashing
        for idx, _ in synsets:
            ids[idx] = self.tokenizer.unk_token_id

        converted_tokens: list = self.tokenizer.convert_ids_to_tokens(ids)

        try:
            final_token_idx = converted_tokens.index(self.tokenizer.eos_token)
            start_at = 0
            if ids[start_at] == self.tokenizer.bos_token_id:
                start_at += 1
            if hasattr(self.tokenizer, 'id_to_lang_code') and ids[start_at].item() in self.tokenizer.id_to_lang_code:
                start_at += 1

            converted_tokens = converted_tokens[start_at:final_token_idx]
        except ValueError as _:
            pass

        for idx, syn_id in synsets:
            ids[idx] = syn_id
            converted_tokens[idx] = self.inverse_label_vocab[syn_id - offset]

        reconstructed = self.tokenizer.convert_tokens_to_string(converted_tokens).replace('▁', ' ')
        # for special_token in self.special_tokens.keys():
        #     reconstructed = reconstructed.replace(special_token, f" {special_token} ")
        return reconstructed.replace('  ', ' ').replace("bn:", " bn:")

    def choose_span_label(self, span: ContextualizedAnnotatedSpan) -> Optional[str]:
        if not self.use_span(span):
            return None

        return span.sense

    def encode_item(self, item: WSDMTItem):
        encodings = []
        positions = [-1]

        for span in item.spans:
            token = span.text
            subword_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
            label = self.choose_span_label(span)
            k = len(subword_ids)
            last_pos = positions[-1]

            for i in range(1, k + 1 + int(label is not None)):
                n = i if label is None else 1
                positions.append(last_pos + n)

            if label is not None:
                encodings.extend(subword_ids)
                encodings.append(self.label_vocab[label] + len(self.tokenizer))
            else:
                encodings.extend(subword_ids)

        encodings.append(self.tokenizer.eos_token_id)
        positions.append(positions[-1] + 1)
        positions.pop(0)  # remove initial -1 element

        max_len = self.tokenizer.model_max_length

        return encodings[:max_len], positions[:max_len]
