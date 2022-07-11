import torch
import torch.nn as nn

from transformers import MarianMTModel

from data.encoder import SentenceBatchEncoder, SenseSentenceBatchEncoder
from model.base import BaseMTModel
from model.sense_embedding import SenseEmbeddingAugmenter


class PatchedEmbedPosition(nn.Module):
    def __init__(self, marian_model: MarianMTModel):
        super().__init__()
        self.original = marian_model.get_encoder().embed_positions
        self._current_position_ids = None

    def set_position_ids(self, position_ids: torch.Tensor):
        self._current_position_ids = position_ids

    def forward(self, *args, **kwargs):
        if self._current_position_ids is None:
            return self.original(*args, **kwargs)
        return nn.Embedding.forward(self.original, self._current_position_ids)


class IntrinsicWSDMTModel(BaseMTModel):
    def __init__(self, encoder: SentenceBatchEncoder,
                 lr=5e-7,
                 warmup_steps=4000,
                 min_lr=1e-9,
                 warmup_init_lr=1e-7,
                 label_smoothing=0.1,
                 pretrained=True,
                 projection_lr=None):
        super().__init__(encoder, lr, warmup_steps, min_lr, warmup_init_lr, label_smoothing, pretrained)
        self.save_hyperparameters('projection_lr')

        self._augmented = isinstance(encoder, SenseSentenceBatchEncoder)

        # only augment the embeddings if we are using senses
        if not self._augmented:
            return

        if 'senses' not in self.hparams:
            encoder: SenseSentenceBatchEncoder
            mapping = encoder.label_vocab
            self.save_hyperparameters(dict(senses=mapping))
        else:
            mapping = self.hparams.senses

        self.model.get_encoder().embed_positions = PatchedEmbedPosition(self.model)

        new_tok_len = len(encoder.tokenizer)
        if new_tok_len != encoder.tokenizer.vocab_size:  # vocab size returns the size without adding tokens
            self.model.resize_token_embeddings(new_tok_len)

        emb = self.model.get_input_embeddings()
        sense_emb = SenseEmbeddingAugmenter(emb, mapping)
        # we just want the sense augmentation in the input embeddings
        # we don't really need to output synsets, do we :) (... for now!)
        self.model.model.encoder.embed_tokens = sense_emb
        # self.model.set_input_embeddings(sense_emb)

    @property
    def sense_embeddings(self) -> SenseEmbeddingAugmenter:
        emb = self.model.model.encoder.embed_tokens
        assert isinstance(emb, SenseEmbeddingAugmenter)
        return emb

    def get_param_groups(self):

        if self.hparams.projection_lr is None or self.hparams.projection_lr == self.hparams.lr:
            return super().get_param_groups()

        architectural_parameters = set(p for p in self.model.parameters())

        param_groups = []

        if self._augmented:
            projection_parameters = set(p for p in self.sense_embeddings.projection.parameters())
            architectural_parameters.difference_update(projection_parameters)
            param_groups.append({
                'params': list(projection_parameters),
                'lr': self.hparams.projection_lr
            })

        param_groups.append({
            'params': list(architectural_parameters),
            'lr': self.hparams.lr
        })

        return param_groups
