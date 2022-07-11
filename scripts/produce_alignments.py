import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--fast_align_path', default='/path/to/fast_align/build')

    return parser.parse_args()


def apply_fast_align(src_it, tgt_it, fast_align_path='/path/to/fast_align/build'):
    with tempfile.TemporaryDirectory() as tempdir:
        directory = Path(tempdir)
        # print(directory)
        piped_file = directory / 'piped.txt'
        with piped_file.open('w') as f_out:
            for l1, l2 in zip(src_it, tgt_it):
                print(f"{l1.rstrip()} ||| {l2.rstrip()}", file=f_out)

        fwd_file = directory / 'fwd.align'
        rev_file = directory / 'rev.align'

        fwd_cmd = f"{fast_align_path}/fast_align -i {piped_file} -d -o -v"
        rev_cmd = f"{fast_align_path}/fast_align -i {piped_file} -d -o -v -r"
        align_cmd = f"{fast_align_path}/atools -i {fwd_file} -j {rev_file} -c grow-diag-final-and"

        # start fast_align of forward and reverse in parallel
        with fwd_file.open('w') as f1, rev_file.open('w') as f2:
            proc_fwd = subprocess.Popen(fwd_cmd.split(), stdout=f1)
            proc_rev = subprocess.Popen(rev_cmd.split(), stdout=f2)

            # wait for them both to finish to launch the symmetrization command
            proc_fwd.wait()
            proc_rev.wait()

            proc_align = subprocess.Popen(align_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, err = proc_align.communicate()
            return out.decode().split('\n')


if __name__ == '__main__':
    args = parse_args()
    with open(args.source) as f_src, open(args.target) as f_tgt:
        print('\n'.join(apply_fast_align(f_src, f_tgt, args.fast_align_path)))
