from argparse import ArgumentParser

import numpy as np
from sacrebleu import corpus_bleu
from tqdm.auto import trange


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('arg3')

    parser.add_argument('--cfr', default=False, action='store_true')
    parser.add_argument('--k', default=0.5, type=float)
    parser.add_argument('--n', default=1000, type=int)

    return parser.parse_args()


def load_sentences(filename):
    with open(filename) as f:
        return [line.rstrip() for line in f]


def bleu(refs, hyps):
    return corpus_bleu(hyps, [refs], force=True).score


def compute_statistic(refs, hyp1, hyp2, k, n):
    c = 0

    if 0 < k <= 1:
        size = round(len(refs) * k)
    else:
        size = int(k)

    indices = np.arange(len(refs))
    it = trange(n, desc=f"Computing BLEUs (k={k}, n={n})")

    for i in it:
        # size = np.random.randint(int(0.1 * len(refs)), int(0.5 * len(refs)))
        sample = np.random.choice(indices, size, replace=True)
        sample1 = [hyp1[i] for i in sample]
        sample2 = [hyp2[i] for i in sample]
        sample_ref = [refs[i] for i in sample]

        bleu1 = bleu(sample_ref, sample1)
        bleu2 = bleu(sample_ref, sample2)
        if bleu1 > bleu2:
            c += 1
        it.set_postfix(dict(c=c, s=c/(i+1)))

    return c / n


def main(args):
    if args.cfr:
        dataset = args.arg3
        exp1_path = f"experiments/{args.arg1}/translations"
        exp2_path = f"experiments/{args.arg2}/translations"
        ref_path = f"{exp1_path}/{dataset}/ref"
        hyp1_path = f"{exp1_path}/{dataset}/hyp"
        hyp2_path = f"{exp2_path}/{dataset}/hyp"
    else:
        ref_path, hyp1_path, hyp2_path = args.arg1, args.arg2, args.arg3

    references = load_sentences(ref_path)
    hyp1 = load_sentences(hyp1_path)
    hyp2 = load_sentences(hyp2_path)
    assert len(references) == len(hyp1) == len(hyp2), f"Different lengths: {len(references)}, {len(hyp1)}, {len(hyp2)}"
    print('BLEU', args.arg1, round(bleu(references, hyp1), 2))
    print('BLEU', args.arg2, round(bleu(references, hyp2), 2))
    print()
    print(compute_statistic(references, hyp1, hyp2, args.k, args.n))


if __name__ == '__main__':
    main(parse_args())
