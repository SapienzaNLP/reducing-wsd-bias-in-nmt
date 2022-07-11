from typing import List, Optional, Union, Dict, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import move_data_to_device

from torch.utils.data import DataLoader

from data.encoder import SentenceBatchEncoder
from data.wsdmt_dataset import WSDMTDataset


def load_few_shot_indices(corpus, split):
    with open(f'data/few-shot/{corpus}/rand_{split}.txt') as f:
        yield from map(int, f)


class WSDMTDataModule(pl.LightningDataModule):
    def __init__(self, encoder: SentenceBatchEncoder, conf):
        super().__init__()
        self.encoder = encoder
        self.src, self.tgt = conf.src, conf.tgt
        self.corpus = conf.corpus_name
        self.variety = conf.dataset_variety
        self.limit = conf.limit if conf.limit else conf.fast_iteration
        self.batch_size = conf.batch_size
        self.eval_batch_size = conf.eval_batch_size
        self.pin_memory = conf.gpus > 0
        self.few_shot = conf.few_shot

        self.datasets: Dict[str, WSDMTDataset] = {}
        self.setup()

    def prepare_data(self, *args, **kwargs):
        # self._get_datasets()
        pass

    @property
    def test_set_names(self):
        return ['test_2014', 'test_2019'] if (self.corpus == 'wmt' and self.few_shot is None) else ['test']

    def _get_normal_datasets(self):
        return ((split, WSDMTDataset(split, self.encoder, self.src, self.tgt,
                                     corpus=self.corpus, variety=self.variety,
                                     limit=self.limit))

                for split in ('train', 'dev', *self.test_set_names))

    def _get_few_shot_datasets(self):
        return ((split.split('_')[0], WSDMTDataset('train', self.encoder, self.src, self.tgt,
                                                   corpus=self.corpus, variety=self.variety,
                                                   identifier=split.split('_')[0],
                                                   limit=load_few_shot_indices(self.corpus, split)))
                for split in (f'train_{self.few_shot}', 'dev', 'test')
                )

    def _get_datasets(self):
        if self.few_shot is not None:
            return self._get_few_shot_datasets()

        return self._get_normal_datasets()

    def setup(self, stage: Optional[str] = None):
        if len(self.datasets) > 0:
            return

        self.datasets = {name: dataset for name, dataset in self._get_datasets()}

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.datasets['train'].loader(batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.datasets['dev'].loader(batch_size=self.eval_batch_size, pin_memory=self.pin_memory)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            self.datasets[split].loader(batch_size=self.eval_batch_size, pin_memory=self.pin_memory)
            for split in self.test_set_names
        ]

    def transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        return move_data_to_device({k: v for k, v in batch.items() if k != 'sids'}, device)
