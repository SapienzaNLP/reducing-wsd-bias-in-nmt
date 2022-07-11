from contextlib import contextmanager
from typing import Optional, List, Set

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from transformers import MarianMTModel, AutoModelForSeq2SeqLM, AutoConfig, MBartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.marian.modeling_marian import shift_tokens_right

from data.encoder import SentenceBatchEncoder
from utils.oom_handler import skip_on_OOM
from utils.scheduler import inverse_sqrt_lr_scheduler
from utils.smoothing import label_smoothed_nll_loss


class BaseMTModel(pl.LightningModule):
    loggable_keys: Set[str] = {'loss', 'xent'}

    def __init__(self, encoder: SentenceBatchEncoder,
                 lr=5e-7,
                 warmup_steps=4000,
                 min_lr=1e-9,
                 warmup_init_lr=1e-7,
                 label_smoothing=0.1,
                 pretrained=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters('lr', 'pretrained', 'warmup_steps', 'min_lr', 'warmup_init_lr', 'label_smoothing')

        model_name = encoder.model_name
        if self.hparams.pretrained:
            self.model: MarianMTModel = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            # config.num_hidden_layers = 3
            # config.encoder_layers = 3
            # config.decoder_layers = 3
            # config.encoder_attention_heads = 4
            # config.decoder_attention_heads = 4
            self.model: MarianMTModel = AutoModelForSeq2SeqLM.from_config(config)

    @property
    def encdec(self):
        return self.model.model

    @property
    def config(self):
        return self.model.config

    @contextmanager
    def fix_position_ids(self, position_ids):
        if position_ids is not None:
            # quirky workaround for well-defined position ids
            # look at IntrinsicWSDModel's __init__() and to PatchedEmbedPosition
            self.model.get_encoder().embed_positions.set_position_ids(position_ids)

        # return control to caller
        yield

        if position_ids is not None:
            self.model.get_encoder().embed_positions.set_position_ids(None)

    def _forward_seq2seq(self,
                         input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         position_ids: Optional[torch.Tensor] = None,
                         sids: Optional[List[str]] = None,
                         labels: Optional[torch.Tensor] = None):

        return_dict = self.config.return_dict
        decoder_input_ids = None
        use_cache = None
        if labels is not None:
            use_cache = False
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        with self.fix_position_ids(position_ids):
            outputs = self.encdec(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=use_cache,
                return_dict=return_dict,
            )

        lm_logits = F.linear(outputs[0], self.encdec.shared.weight, bias=self.model.final_logits_bias)

        nll_loss = None
        xent = None
        loss = None
        if labels is not None:
            smoothing = self.hparams.label_smoothing
            if smoothing > 0:
                xent = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size),
                                       labels.view(-1),
                                       ignore_index=self.config.pad_token_id).item()
                # get log probabilities
                lprobs = F.log_softmax(lm_logits, dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(lprobs.view(-1, self.config.vocab_size), labels.view(-1),
                                                         epsilon=smoothing, ignore_index=self.config.pad_token_id)

            else:

                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        out = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

        out['last_hidden_state'] = outputs.last_hidden_state

        if nll_loss is not None:
            out['nll_loss'] = nll_loss
            out['xent'] = xent

        return out

    @skip_on_OOM()
    def backward(self, *args, **kwargs) -> None:
        return super().backward(*args, **kwargs)

    @skip_on_OOM()
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                sids: Optional[List[str]] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):
        seq2seq_output = self._forward_seq2seq(input_ids=input_ids, attention_mask=attention_mask,
                                               position_ids=position_ids, labels=labels)
        return seq2seq_output

    def _shared_step(self, batch, prefix=None):
        result = self(**batch)
        if result is None:
            return {}

        prefix = f"{prefix}_" if prefix is not None else ""
        ret = {
            f"{prefix}{k}": v
            for k, v in result.items()
            if k in self.loggable_keys
        }
        self.log_dict(ret)
        return ret

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch).get('loss', None)

    def _shared_eval(self, batch, prefix):
        return self._shared_step(batch, prefix)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_eval(batch, 'test')

    def get_param_groups(self):
        return self.model.parameters()

    def configure_optimizers(self):
        optimizer = AdamW(self.get_param_groups(), lr=self.hparams.lr,
                          betas=(0.9, 0.98), weight_decay=0.0)
        scheduler = inverse_sqrt_lr_scheduler(optimizer,
                                              warmup_steps=self.hparams.warmup_steps,
                                              base_lr=self.hparams.lr,
                                              warmup_init_lr=self.hparams.warmup_init_lr,
                                              min_lr=self.hparams.min_lr)

        return dict(optimizer=optimizer,
                    lr_scheduler=dict(
                        scheduler=scheduler,
                        interval='step',
                        frequency=1,
                    ))
