import random

from typing import Iterator, Sized, Sequence, List

import torch
import numpy as np

from torch.utils.data import Sampler


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


class MaxTokensBatchSampler(Sampler[Sequence[int]]):
    def __init__(self,
                 data_source: Sized,
                 max_tokens: int = 4096,
                 noise_value: float = 0.1):
        super().__init__(data_source)
        self.sizes = np.array(data_source)
        self.max_tokens = max_tokens
        self._noise_value = noise_value
        self._noisy_indices = self._get_noisy_indices()

    def _get_noisy_indices(self):
        return self.noisy_argsort(self.sizes, self._noise_value)

    @staticmethod
    def noised(array: np.ndarray, noise_value: float = 0.1) -> np.ndarray:
        noise = 1 + np.random.uniform(-noise_value, noise_value, array.shape)
        noised_array = array * noise
        return noised_array

    @staticmethod
    def noisy_argsort(array: np.ndarray, noise_value: float = 0.1) -> np.ndarray:
        return MaxTokensBatchSampler.noised(array, noise_value).argsort()

    def _lazy_groups_of_max_size(self, max_size: int = None):
        cur_max_size = 0
        group: List[int] = []
        max_size = max_size or self.max_tokens

        for index in self._noisy_indices:
            size = self.sizes[index]

            # if size > self.max_tokens:
            #     logger.warning(
            #         "Found instance of size %d, which is bigger than the expected size for a batch (%d)",
            #         size,
            #         self.max_tokens,
            #     )
            group_size = max(size, cur_max_size) * (len(group) + 1)

            if group_size > max_size:
                yield group
                cur_max_size = 0
                group = []

            group.append(index)
            cur_max_size = max(cur_max_size, size)

    def __iter__(self) -> Iterator[Sequence[int]]:
        yield from self._lazy_groups_of_max_size()

    def __len__(self):
        return sum(1 for _ in self)


if __name__ == '__main__':
    random.seed(42)
    torch.random.manual_seed(42)

    sizes = torch.randint(1, 512, (1000000,)).tolist()
    bs = MaxTokensBatchSampler(sizes)
    pad_wasted = []

    for i, batch_indices in enumerate(bs):
        elements = bs.sizes[batch_indices].tolist()
        tensor_size = len(elements) * max(elements)
        tokens_size = sum(elements)
        padding_ratio = 1 - tokens_size / tensor_size
        pad_wasted.append(padding_ratio)

        if tokens_size > bs.max_tokens:
            print(f"{i})")
            print(f"\t- elements: {batch_indices}")
            print(f"\t- sizes: {elements}")
            print(f"\t- numbers: {tensor_size} tensor size, {tokens_size} actual tokens ({padding_ratio * 100:.1f}% pad)")

    print(f"We have a total of {len(bs)} batches")
    print(f"Wasted {np.mean(pad_wasted) * 100:.4f}% on average on padding")


