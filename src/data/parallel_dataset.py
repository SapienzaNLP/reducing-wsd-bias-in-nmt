from abc import ABC

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer
import torch


class MTDataset(Dataset, ABC):
    src_lang: str
    tgt_lang: str
    name: str
    tokenizer: PreTrainedTokenizer

    def decode(self, ids):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)

        assert len(ids.shape) == 1

        return self.tokenizer.decode(ids, skip_special_tokens=True).replace('▁', ' ').replace('  ', ' ')

    @property
    def src(self):
        return self.src_lang

    @property
    def tgt(self):
        return self.tgt_lang

    @property
    def langs(self):
        return self.src, self.tgt

