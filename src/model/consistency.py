from typing import Set, Optional, List, Any, Dict

import torch
from model.intrinsic import IntrinsicWSDMTModel
from utils.oom_handler import skip_on_OOM


class ConsistencyRegularizationModel(IntrinsicWSDMTModel):
    loggable_keys: Set[str] = {'loss', 'xent', 'base_loss', 'enhanced_loss', 'kl_div'}

    kl = torch.nn.KLDivLoss(reduction='batchmean')
    softmax = torch.nn.Softmax(dim=-1)
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    reg = 0.5
    # tau = 1

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = checkpoint['state_dict']
        s = 'model.model.encoder'
        if (pos_weight := state.pop(f'{s}.embed_positions.weight', None)) is not None:
            state[f'{s}.embed_positions.original.weight'] = pos_weight

        if (emb_weight := state.pop(f'{s}.embed_tokens.weight', None)) is not None:
            state[f'{s}.embed_tokens._base_embedding.weight'] = emb_weight

    @skip_on_OOM()
    def forward(self,
                input_ids: torch.Tensor,
                enhanced_input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                enhanced_attention_mask: Optional[torch.Tensor] = None,
                enhanced_position_ids: Optional[torch.Tensor] = None,
                sids: Optional[List[str]] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):

        base_pass = super().forward(input_ids, attention_mask=attention_mask, labels=labels)
        enhanced_pass = super().forward(enhanced_input_ids, attention_mask=enhanced_attention_mask,
                                        position_ids=enhanced_position_ids, labels=labels)

        base_loss, enhanced_loss = base_pass.loss, enhanced_pass.loss
        base_logits, enhanced_logits = base_pass.logits, enhanced_pass.logits
        base_distribution = self.softmax(base_logits)
        enhanced_distribution = self.log_softmax(enhanced_logits).detach()
        kl_div = self.kl(enhanced_distribution, base_distribution)

        loss = 0.5 * base_loss + 0.5 * enhanced_loss + self.reg * kl_div

        return dict(
            base_loss=base_loss.item(),
            enhanced_loss=enhanced_loss.item(),
            kl_div=kl_div.item(),
            loss=loss
        )
