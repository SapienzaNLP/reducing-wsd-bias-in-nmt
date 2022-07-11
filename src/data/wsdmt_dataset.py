from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.encoder import SentenceBatchEncoder
from data.entities import WSDMTParallelItem
from data.parallel_dataset import MTDataset


def map_lengths(dataset, tokenizer, langs):
    # NOTE: assumes langs are in src, tgt order
    lang1, lang2 = langs
    tok_name = tokenizer.__class__.__name__.replace('Tokenizer', '')

    def fn(_, index):
        sentence1 = dataset[index][lang1]['sentence']
        sentence2 = dataset[index][lang2]['sentence']
        encoding = tokenizer.prepare_seq2seq_batch(
            src_texts=sentence1, tgt_texts=sentence2,
            src_lang=lang1, tgt_lang=lang2,
        )
        return {
            f'{lang1}_length': len(encoding['input_ids']),
            f'{lang2}_length': len(encoding['labels'])
        }

    return map_named(dataset, fn, f'with_{tok_name}_lengths', dict(with_indices=True))


def map_named(dataset, fn, identifier, map_kwargs=None):
    map_kwargs = map_kwargs or {}

    cache_name = f"{dataset.config_name}@{dataset.split}-{identifier}"
    cache_file_name = dataset._get_cache_file_path(cache_name)
    # print(cache_file)
    num_proc = None
    suffix_template = None
    if dataset.split == 'train':
        num_proc = 16
        suffix_template = "_{rank:d}"

    map_kwargs['cache_file_name'] = cache_file_name
    map_kwargs['num_proc'] = num_proc
    map_kwargs['suffix_template'] = suffix_template

    return dataset.map(fn, **map_kwargs)


class WSDMTDataset(MTDataset):
    def __init__(self, split, encoder: SentenceBatchEncoder,
                 src='en', tgt='de', corpus='wmt', variety='all',
                 limit=None, compute_lengths=False,
                 identifier=None, use_dataset=True):

        challenge = False
        if split.startswith('challenge'):
            split = split.split('@')[1]
            challenge = True

        self.src_lang, self.tgt_lang, self.corpus, self.split = src, tgt, corpus, split
        self.name = identifier or split

        encoder.link(self)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder.model_name, use_fast=False)
        if use_dataset:
            _dataset = load_dataset('Valahaar/wsdmt', corpus=corpus, variety=variety,
                                    lang1=src, lang2=tgt, split=split, challenge=challenge)
            self.dataset = map_lengths(_dataset, self.tokenizer, self.langs) if compute_lengths else _dataset

        self.limit = limit

        if limit and use_dataset:
            if isinstance(limit, int):
                limit = range(limit)
            self.dataset = self.dataset.select(limit)

        if use_dataset:
            self.sids = {sid: i for i, sid in enumerate(self.dataset['sid'])}
            self.sizes = {lang: self.dataset[f'{lang}_length'] for lang in self.langs} if compute_lengths else {}

        self.encoder = encoder
        encoder.fit(self)

    def __getitem__(self, item) -> 'WSDMTParallelItem':
        idx = self.sids[item] if isinstance(item, str) else int(item)
        return WSDMTParallelItem(self, idx)

    def __iter__(self):
        return map(self.__getitem__, range(len(self)))

    def __len__(self):
        return len(self.sids)

    def loader(self, batch_size, shuffle=False, pin_memory=True, num_workers=16):
        return DataLoader(self,
                          batch_size=batch_size, shuffle=shuffle,
                          # batch_sampler=MaxTokensBatchSampler(self.sizes[self.src]),
                          pin_memory=pin_memory, num_workers=num_workers,
                          collate_fn=self.encoder.encode)


# if __name__ == '__main__':
#     from tqdm.auto import tqdm
#
#     src = 'en'
#     tgt = 'de'
#     corpus = 'wmt'
#     split = 'dev'
#
#     ds = WSDMTDataset(src, tgt, corpus, 'train')
#     device = torch.device('cuda')
#     loader = ds.loader(0, pin_memory=True, num_workers=16)
#     for i, batch in enumerate(tqdm(loader)):
#         move_data_to_device(batch, device)
#         x = (batch['input_ids'] != ds.tokenizer.pad_token_id).sum().item()
#         if x > 4096:
#             print(i, batch)
