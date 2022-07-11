from typing import Dict

import torch
import torch.nn as nn


def load_ares_embeddings(synsets: Dict[str, int]) -> nn.Embedding:

    print(f"Loading ARES embeddings for {len(synsets)} synsets")

    ares, syn_mapping = torch.load(f"data/ares.pt")
    matrix = torch.empty(len(synsets), ares.shape[1])

    for synset, idx in synsets.items():
        if synset == 'NOSENSE':
            matrix[idx] = torch.zeros_like(matrix[0])
            continue
        ares_idx = syn_mapping[synset]
        matrix[idx] = ares[ares_idx]

    return nn.Embedding.from_pretrained(matrix)


class SenseEmbeddingAugmenter(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 synset_mapping: Dict[str, int]):
        super().__init__()

        self._base_embedding = embedding
        self._sense_embedding = self._load_embedding(synset_mapping)
        self._sense_embedding.requires_grad_(False)
        # self._sense_embedding.reset_parameters()
        self._vocab_size = embedding.num_embeddings

        self.projection = nn.Sequential(
            self._sense_embedding,
            nn.Linear(self._sense_embedding.embedding_dim, embedding.embedding_dim, False),
            # nn.ReLU()
        )

    @property
    def weight(self):
        return self._base_embedding.weight

    def forward(self, input_ids: torch.Tensor):
        offset = self._vocab_size

        # mask with True corresponding to original tokens (not sense ids)
        orig_pos_mask = (input_ids < offset)
        if orig_pos_mask.all():  # all true -> no synset -> we can return the base embedding
            return self._base_embedding(input_ids)

        # keep only input_ids which are from original
        base_emb_ids = (input_ids * orig_pos_mask)
        # mask out input_ids which
        sense_emb_ids = ((input_ids - offset) * ~orig_pos_mask)

        orig_pos_mask = orig_pos_mask.unsqueeze(-1)
        original_vectors = self._base_embedding(base_emb_ids)
        sense_vectors = self.projection(sense_emb_ids)
        masked_original_vectors = original_vectors * orig_pos_mask
        masked_sense_vectors = sense_vectors * ~orig_pos_mask
        final_vectors = masked_original_vectors + masked_sense_vectors
        return final_vectors

    @classmethod
    def _load_embedding(cls, mapping) -> nn.Embedding:
        # return nn.Embedding(10, 6)
        embedding = load_ares_embeddings(mapping)
        return embedding


if __name__ == '__main__':
    emb = nn.Embedding(10, 5)
    mapping = {}
    sense_emb = SenseEmbeddingAugmenter(emb, mapping)
    ids = torch.tensor([[3, 4, 8, 12, 0]])
    print(sense_emb(torch.tensor([[3, 4, 8, 12, 0]])))
    print(sense_emb(torch.tensor([[3, 4, 8, 9, 0]])))
