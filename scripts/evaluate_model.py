import os
import subprocess
import sys
from argparse import ArgumentParser

from more_itertools import chunked
from tqdm.auto import tqdm

from data.encoder import ENCODERS, get_encoder, PROVIDERS, get_provider
from data.encoder.model_provider import ModelProvider, MBart50ModelProvider
from data.wsdmt_dataset import WSDMTDataset
from model.base import BaseMTModel
from model.consistency import ConsistencyRegularizationModel
from model.disambiguator import EncoderDisambiguationMTModel, DecoderDisambiguationMTModel
from model.intrinsic import IntrinsicWSDMTModel
from model.sense_embedding import SenseEmbeddingAugmenter
from scripts.produce_alignments import apply_fast_align
from utils.translation_system import TranslationSystem


def choose_model_class(encoder_type):
    if encoder_type == 'dis_enc':
        return EncoderDisambiguationMTModel

    if encoder_type == 'dis_dec':
        return DecoderDisambiguationMTModel

    # return ConsistencyRegularizationModel
    return IntrinsicWSDMTModel


def parse_args():
    parser = ArgumentParser()

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--exp')
    grp.add_argument('--baseline')

    parser.add_argument('--encoder-type', default='basic', choices=ENCODERS)
    parser.add_argument('--provider', default='opus', choices=PROVIDERS)
    parser.add_argument('--ckpt', default='best.ckpt')

    parser.add_argument('--bleu', action='store_true')

    parser.add_argument('--num-beams', default=5, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sense-based', action='store_true')

    return parser.parse_args()


def on_load_checkpoint(self, checkpoint):
    # sd = self.state_dict()
    csd = checkpoint['state_dict']
    s = 'model.model.encoder'
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


def main(args):
    exp_dir = f'experiments/{args.exp}'
    langs, corpus, exp_name = args.exp.split('/')
    src, tgt = langs.split('-')

    provider = get_provider(args.provider)()
    encoder = get_encoder('sense' if args.sense_based else args.encoder_type)(provider)
    dev = WSDMTDataset('dev', variety='all', src=src, tgt=tgt, corpus=corpus, encoder=encoder)
    # model = BaseMTModel(encoder=encoder)
    cls = choose_model_class(args.encoder_type)
    cls.on_load_checkpoint = on_load_checkpoint

    if args.sense_based:
        cls.on_load_checkpoint = fix_encoder_on_load_checkpoint(encoder)

    model = cls.load_from_checkpoint(f'{exp_dir}/{args.ckpt}', encoder=encoder)
    model.to(args.device).freeze()
    system = TranslationSystem(model, encoder, args.num_beams)

    # ds_names = ('test_2014', 'test_2019')
    ds_names = ('dev', 'test')

    if args.bleu:
        print(f'evaluation for {args.exp}')
        for name in ds_names:
            ds = WSDMTDataset(name, variety='all', src=src, tgt=tgt, corpus=corpus, encoder=encoder)
            if args.sense_based:
                print(system.evaluate_loader(ds.loader(args.batch_size, pin_memory=False), name))
            else:
                print(system.evaluate_dataset(ds, batch_size=args.batch_size))
        return

    def mkdir(directory):
        if not os.path.exists(directory):
            os.mkdir(directory)

    def count_lines(filename):
        if not os.path.exists(filename):
            return -1

        with open(filename) as f:
            return sum(1 for line in f if len(line.rstrip()) > 0)

    def check_completeness(infile, outfile):
        n1 = count_lines(infile)
        n2 = count_lines(outfile)
        return n1 == n2 and n2 > 0

    def translate(infile, outfile):
        if check_completeness(infile, outfile):
            print(f"{infile} and {outfile} have the same number of lines, skipping translation")
        else:
            with open(infile) as f_in, open(outfile, 'w') as f_out:
                iterator = tqdm(map(lambda e: e.rstrip(), f_in), desc=f'Translating {infile}')
                for lines in chunked(iterator, args.batch_size):
                    translations = system.translate(lines)
                    print(*translations, sep='\n', file=f_out)

    def align(infile, outfile):
        if check_completeness(outfile, outfile + '.align'):
            print(f'alignments for {outfile} already computed, skipping')
        else:
            with open(infile) as f_in, open(outfile) as f_out, open(outfile + '.align', 'w') as f_align:
                alignments = apply_fast_align(f_in, f_out)
                print(*alignments, sep='\n', file=f_align)

    def translate_and_align(infile, outfile):
        translate(infile, outfile)
        align(infile, outfile)

    challenge_dir = f"nmt-data/{corpus}/challenge"
    translations_dir = f"{exp_dir}/translations"
    mkdir(translations_dir)
    mkdir(f"{translations_dir}/wsd_bias")
    mkdir(f"{translations_dir}/adversarial")

    # copy existing data
    for lang in (src, tgt):
        os.system(f'cp {challenge_dir}/wsd_bias/{lang} {translations_dir}/wsd_bias/{lang}')
        os.system(f'cp {challenge_dir}/adversarial/{lang} {translations_dir}/adversarial/{lang}')

    os.system(f'cp {challenge_dir}/adversarial/adv.{src} {translations_dir}/adversarial/adv.{src}')
    sense_clusters_path = "data/wsd_bias/ende_homograph_sense_clusters.json"
    attractors_path = f"data/wsd_bias/attractors/{corpus}_attractors.json"

    base_cmd = f"{sys.executable} scripts/evaluate_attack_success.py " \
               f"--lang_pair {langs} " \
               f"--output_dir {translations_dir}/{{challenge}}/output " \
               f"--attractors_path {attractors_path} " \
               f"--json_challenge_set_path " \
               f"data/wsd_bias/json_challenge_sets/{corpus}_{{challenge}}_challenge_set.json " \
               f"--sense_clusters_path {sense_clusters_path} "

    wsd_bias_cmd = f"--source_sentences_path {translations_dir}/{{challenge}}/{src} " \
                   f"--translations_path {translations_dir}/{{challenge}}/pred.{tgt} " \
                   f"--alignments_path {translations_dir}/{{challenge}}/pred.{tgt}.align "

    adversarial_cmd = f"--adversarial_source_sentences_path {translations_dir}/{{challenge}}/adv.{src} " \
                      f"--adversarial_translations_path {translations_dir}/{{challenge}}/pred.adv.{tgt} " \
                      f"--adversarial_alignments_path {translations_dir}/{{challenge}}/pred.adv.{tgt}.align "

    # translate and align nmt-data/{corpus}/challenge/{challenge}/[en,adv.en] ->
    #       experiments/{langs}/{corpus}/{exp}/translations/{challenge}/[de,adv.de]<.align>
    translate_and_align(f"{translations_dir}/wsd_bias/{src}", f"{translations_dir}/wsd_bias/pred.{tgt}")

    wsd_bias_proc = subprocess.Popen((base_cmd + wsd_bias_cmd).format(challenge='wsd_bias').split())

    translate_and_align(f"{translations_dir}/adversarial/{src}", f"{translations_dir}/adversarial/pred.{tgt}")
    translate_and_align(f"{translations_dir}/adversarial/adv.{src}", f"{translations_dir}/adversarial/pred.adv.{tgt}")
    adversarial_proc = subprocess.Popen((base_cmd + wsd_bias_cmd + adversarial_cmd)
                                        .format(challenge='adversarial').split())

    wsd_bias_proc.wait()
    adversarial_proc.wait()


if __name__ == '__main__':
    main(parse_args())
