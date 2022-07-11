import fileinput
import os
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from data.encoder import ENCODERS, PROVIDERS, get_provider, get_encoder
from data.wsdmt_dataset import WSDMTDataset
from model.base import BaseMTModel
from model.intrinsic import IntrinsicWSDMTModel
from model.sense_embedding import SenseEmbeddingAugmenter
from utils.translation_system import TranslationSystem


def on_load_checkpoint(self, checkpoint):
    csd = checkpoint['state_dict']
    s = 'model.model.encoder'

    if f'{s}.embed_tokens._sense_embedding.weight' not in csd:
        return

    csd[f'{s}.embed_tokens.weight'] = csd.pop(f'{s}.embed_tokens._base_embedding.weight')
    csd[f'{s}.embed_positions.weight'] = csd.pop(f'{s}.embed_positions.original.weight')
    csd.pop(f'{s}.embed_tokens._sense_embedding.weight')
    csd.pop(f'{s}.embed_tokens.projection.0.weight')
    csd.pop(f'{s}.embed_tokens.projection.1.weight')


def fix_encoder_on_load_checkpoint(encoder):
    def on_load_checkpoint_sense_based(self, checkpoint):
        model_encoder = self.model.model.encoder
        senses = checkpoint['hyper_parameters']['senses']
        model_encoder.embed_tokens = SenseEmbeddingAugmenter(model_encoder.embed_tokens._base_embedding, senses)
        encoder.senses_mapping = senses

    return on_load_checkpoint_sense_based


def load_sentences(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f]


def dump_sentences(sentences, filename):
    with open(filename, 'w') as f:
        for sentence in sentences:
            print(sentence, file=f)


def count_lines(filename):
    if not os.path.exists(filename):
        return -1

    with open(filename) as f:
        return sum(1 for line in f if len(line.rstrip()) > 0)


def check_same_length(infile, outfile):
    n1 = count_lines(infile)
    n2 = count_lines(outfile)
    return n1 == n2 and n2 > 0


def main(args):
    exp_dir = Path(f'experiments') / args.exp
    langs, corpus, exp_name = args.exp.split('/')
    src, tgt = langs.split('-')

    provider = get_provider(args.provider)()
    encoder = get_encoder(args.encoder_type)(provider)

    # needed to populate the encoder
    WSDMTDataset(split='dev', encoder=encoder, src=src, tgt=tgt, corpus=corpus, variety=args.variety, use_dataset=False)

    if args.baseline:
        model = BaseMTModel(encoder=encoder)
    else:
        IntrinsicWSDMTModel.on_load_checkpoint = on_load_checkpoint
        model = IntrinsicWSDMTModel.load_from_checkpoint(f'{exp_dir}/{args.ckpt}', encoder=encoder)

    system = TranslationSystem(model, encoder, args.beam_size)
    model.to(args.device).freeze()

    if args.datasets is not None:
        datasets = args.datasets.split(',')
        ds_it = tqdm(datasets, desc="Translating datasets")

        for dataset in ds_it:
            translations_dir = exp_dir / 'translations' / dataset
            translations_dir.mkdir(parents=True, exist_ok=True)

            tgt_file = translations_dir / 'ref'
            pred_file = translations_dir / 'hyp'

            if check_same_length(tgt_file, pred_file) and not args.force:
                continue

            if dataset == 'tatoeba':
                src_sents, tgt_sents = system.tatoeba.sentences, system.tatoeba.references
            else:
                dataset_folder = f'nmt-data/{corpus}/{dataset}'
                src_sents = load_sentences(f"{dataset_folder}/{src}")
                tgt_sents = load_sentences(f"{dataset_folder}/{tgt}")

            dump_sentences(tgt_sents, tgt_file)
            pred_sents = system.translate_corpus(src_sents, batch_size=args.batch_size)
            pred_it = tqdm(pred_sents, desc=f"Translating {dataset}", position=1, leave=False, total=len(src_sents))
            dump_sentences(pred_it, pred_file)

    else:
        with fileinput.input(args.input) as f_in:
            for translation in system.translate_corpus(f_in, batch_size=args.batch_size):
                print(translation)


def parse_args():
    parser = ArgumentParser()

    parser.add_mutually_exclusive_group()
    parser.add_argument('--input')
    parser.add_argument('--datasets', default='tatoeba,test')

    parser.add_argument('--exp', required=True)
    parser.add_argument('--ckpt', default='best.ckpt')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--force', action='store_true')

    parser.add_argument('--encoder-type', default='basic', choices=ENCODERS)
    parser.add_argument('--provider', default='opus', choices=PROVIDERS)
    parser.add_argument('--variety', default='all')

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--device', default='cuda')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
