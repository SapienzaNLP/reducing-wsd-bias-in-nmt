import os
from pathlib import Path
from typing import Tuple, Iterable, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from wandb.sdk.wandb_run import Run

from data.datamodule import WSDMTDataModule
from data.wsdmt_dataset import WSDMTDataset
from model.intrinsic import IntrinsicWSDMTModel
from utils.translation_system import TranslationSystem


class SubwordsMonitor(pl.Callback):
    def __init__(self, pad_token_id):
        self._pad_token_id = pad_token_id
        self._total = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        n_subwords_batch = (batch['input_ids'] != self._pad_token_id).sum().item()
        self._total += n_subwords_batch
        pl_module.log('subwords', self._total, prog_bar=True, on_step=True, on_epoch=False)


class FullEvaluationCallback(pl.Callback):
    NamedDataLoader = Tuple[str, DataLoader]

    @classmethod
    def from_datamodule(cls, translator: TranslationSystem, module: WSDMTDataModule, test_after_step: int = 4000):

        datasets = [module.datasets[name] for name in module.test_set_names]
        loaders = list(zip(module.test_set_names, module.test_dataloader()))

        return FullEvaluationCallback(translator,
                                      module.datasets['dev'],
                                      module.val_dataloader(),
                                      test_after_step,
                                      datasets,
                                      loaders)

    def __init__(self,
                 translator: TranslationSystem,
                 dev_dataset: WSDMTDataset,
                 dev_loader: DataLoader,
                 test_after_global_step: int = 4000,
                 test_dataset: Union[WSDMTDataset, Iterable[WSDMTDataset]] = None,
                 test_loader: Optional[Union[NamedDataLoader, Iterable[NamedDataLoader]]] = None):

        self._translator: TranslationSystem = translator

        self.test_after_step = test_after_global_step

        self.dev_loader = dev_loader
        self.dev_dataset = dev_dataset

        self.test_dataset = test_dataset if isinstance(test_dataset, (tuple, list)) else (test_dataset,)
        self.test_loader = test_loader

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: IntrinsicWSDMTModel):
        if trainer.fast_dev_run or trainer.running_sanity_check:
            return

        experiment: Run = trainer.logger.experiment

        evaluations = [
            self._translator.evaluate_dataset(self.dev_dataset),
            self._translator.evaluate_loader(self.dev_loader, 'dev')
        ]

        if trainer.global_step > self.test_after_step:

            for ds in self.test_dataset:
                dataset_eval = self._translator.evaluate_dataset(ds)
                evaluations.append(dataset_eval)

            for loader in (self.test_loader or []):
                loader_eval = self._translator.evaluate_loader(loader[1], loader[0])
                evaluations.append(loader_eval)

        print()
        print(f"Evaluations at global step {trainer.global_step}, epoch {trainer.current_epoch}:")
        for corpus in evaluations:
            if corpus is None:
                continue
            print(f'\t{corpus.name}: {corpus.metrics.metrics_str()}')

            to_log = {
                f"{corpus.name}_{metric}": score
                for metric, score in corpus.metrics
            }

            pl_module.log_dict(to_log)

            # only log for wandb experiment (in --offline mode it's a Tensorboard SummaryWriter)
            if not isinstance(experiment, Run):
                continue

            experiment.log({corpus.name: corpus.wandb_table()}, commit=False)

        print()


class LinkBestModelCheckpoint(ModelCheckpoint):
    CHECKPOINT_NAME_BEST = 'best.ckpt'

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        save_dir = Path(self.dirpath)
        save_dir_fd = os.open(save_dir, os.O_RDONLY)
        if self.best_model_path != "":
            orig_best = Path(self.best_model_path)
            save_dir = orig_best.parent
            (save_dir / self.CHECKPOINT_NAME_BEST).unlink(missing_ok=True)
            os.symlink(orig_best.name, self.CHECKPOINT_NAME_BEST, dir_fd=save_dir_fd)

        if self.last_model_path != "":
            orig_last = Path(self.last_model_path)
            (save_dir / ModelCheckpoint.CHECKPOINT_NAME_LAST).unlink(missing_ok=True)
            os.symlink(orig_last.name, ModelCheckpoint.CHECKPOINT_NAME_LAST, dir_fd=save_dir_fd)

        os.close(save_dir_fd)
