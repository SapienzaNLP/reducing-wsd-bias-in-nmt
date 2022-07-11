from typing import List, TYPE_CHECKING, Optional

import torch

from data.encoder.encoder import SentenceBatchEncoder

if TYPE_CHECKING:
    from data.wsdmt_dataset import WSDMTParallelItem, WSDMTDataset


class BasicSentenceBatchEncoder(SentenceBatchEncoder):
    def _fit(self, dataset: 'WSDMTDataset'):
        pass

    def encode_sentences(self, src_sentences: List[str], tgt_sentences: Optional[List[str]] = None):
        src, tgt = self.langs

        if hasattr(self.tokenizer, 'lang_code_to_id'):
            src = next(k for k in self.tokenizer.lang_code_to_id if k[:2] == src)
            tgt = next(k for k in self.tokenizer.lang_code_to_id if k[:2] == tgt)

        return self.tokenizer.prepare_seq2seq_batch(src_texts=src_sentences,
                                                    tgt_texts=tgt_sentences,
                                                    src_lang=src,
                                                    tgt_lang=tgt if tgt_sentences is not None else None,
                                                    return_tensors='pt')

    def encode(self, items: List['WSDMTParallelItem']):
        sids, src_sents, tgt_sents = zip(*((item.sid, item.src_item.sentence, item.tgt_item.sentence)
                                           for item in items))
        batch = self.encode_sentences(src_sents, tgt_sents)
        batch['sids'] = sids
        return batch

    def decode_ids(self, ids):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)

        assert len(ids.shape) == 1

        return self.tokenizer.decode(ids, skip_special_tokens=True).replace('▁', ' ').replace('  ', ' ').strip()
