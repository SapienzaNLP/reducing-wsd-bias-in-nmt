from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data.datamodule import WSDMTDataModule
from data.encoder import ENCODERS, get_encoder, PROVIDERS, get_provider
from model.consistency import ConsistencyRegularizationModel
from utils.callbacks import FullEvaluationCallback, SubwordsMonitor, LinkBestModelCheckpoint
from model.intrinsic import IntrinsicWSDMTModel
from utils.translation_system import TranslationSystem


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--experiment', default='unnamed-exp')
    parser.add_argument('--src', default='en')
    parser.add_argument('--tgt', default='it')
    parser.add_argument('--encoder-type', default='basic', choices=ENCODERS)
    parser.add_argument('--limit', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--no-pretraining', dest='pretrained', action='store_false')
    parser.add_argument('--tags', type=str, default=None)  # comma separated (i.e. --tags run,dev,xyz)
    parser.add_argument('--corpus-name', type=str, default=None)
    parser.add_argument('--dataset-variety', type=str, default=None, required=True)
    parser.add_argument('--offline', default=False, action='store_true')
    parser.add_argument('--fast-iteration', default=False, action='store_true')
    parser.add_argument('--scr', '--consistency-regularization', dest='scr', default=False, action='store_true')
    parser.add_argument('--restart-from', default=None)
    parser.add_argument('--provider', default='opus', choices=PROVIDERS)

    parser.add_argument('--few-shot', default=None, choices=['1k', '10k', '50k'])
    parser.add_argument('--fine-tune', default=False, action='store_true')

    # model parameters
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--projection-lr', type=float, default=None)

    # loss label smoothing
    parser.add_argument('--smoothing', type=float, default=0.0)

    # scheduler params
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--warmup-init-lr', type=float, default=1e-7)
    parser.add_argument('--min-lr', type=float, default=1e-9)

    # data parameters
    # parser.add_argument('--data-path', default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--virtual-batch-size', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--eval-batch-multiplier', type=int, default=2)

    # early stopping / checkpointing
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor', type=str, default='dev_bleu')
    parser.add_argument('--monitor-mode', type=str, default='max', choices=['min', 'max', 'auto'])
    parser.add_argument('--save-top', type=int, default=5)

    parser = pl.Trainer.add_argparse_args(parser)

    return parser.parse_args()


def validate_args(args: Namespace):
    # restart_from is used for consistency regularization
    # resume / resume_from are for resuming training
    # assert not args.restart_from and (args.resume or args.resume_from)

    if args.fast_iteration:
        args.fast_iteration = 500_000

    if args.limit:
        args.limit = 1000

    vbs = args.virtual_batch_size
    bs = args.batch_size

    if vbs:
        assert vbs % bs == 0, "--virtual-batch-size must be a multiple of --batch-size"
        args.accumulate_grad_batches = args.virtual_batch_size // args.batch_size

    ebs = args.eval_batch_size
    ebm = args.eval_batch_multiplier

    if ebm:
        assert not ebs, "cannot set both --eval-batch-size and --eval-batch-multiplier"
        args.eval_batch_size = bs * ebm

    if not args.max_epochs and not args.few_shot:
        args.max_epochs = 20

    # if args.few_shot:
    #     args.check_val_every_n_epoch = 5

    return args


def tags_from_args(args):
    tags = [
        f"{args.src}-{args.tgt}",
        args.corpus_name,
        args.dataset_variety,
        args.encoder_type
    ]

    if args.tags is not None:
        tags.extend(args.tags.split(','))

    tags.append(args.encoder_type)

    if args.limit:
        tags.append('limited')

    if args.fast_iteration:
        tags.append('fast-iteration')

    if args.smoothing > 0:
        tags.append(f'smoothing@{args.smoothing:.2f}')

    if args.few_shot:
        tags.append(f'few-shot@{args.few_shot}')

    return tags


def main():
    args = validate_args(parse_args())
    # args.limit = 200
    # args.experiment = 'test-few-shot'
    # args.smoothing = 0.1
    # args.gpus = 0
    # args.seed = 1
    # args.data_path = 'nmt-data'
    # args.offline = True

    args.seed = pl.seed_everything(args.seed)

    exp_name = f"{args.src}-{args.tgt}/{args.corpus_name}/{args.experiment}"

    folder = Path(f"experiments/{exp_name}")
    folder.mkdir(exist_ok=True, parents=True)

    provider = get_provider(args.provider)()
    encoder = get_encoder(args.encoder_type)(provider)
    if args.scr:
        encoder.output_prefix = 'enhanced_'

    module = WSDMTDataModule(encoder, args)

    resume_from = args.resume_from

    tags = tags_from_args(args)
    model_class = ConsistencyRegularizationModel if args.scr else IntrinsicWSDMTModel

    if (args.scr or args.fine_tune) and (args.restart_from or args.resume):
        if args.resume:
            if resume_from is None:
                checkpoints = filter(lambda file: file.name.endswith('.ckpt'), folder.iterdir())
                resume_from = str(sorted(checkpoints, key=lambda file: file.stat().st_ctime, reverse=True)[0])
                print('Resuming from last checkpoint:', resume_from)
            else:
                print('Resuming from', resume_from)
            ckpt_path = resume_from
            ckpt = torch.load(ckpt_path)
            assert encoder.senses_mapping == ckpt['hyper_parameters']['senses'], (
                'encoder mapping is not the same as stored one:\n'
                f'{encoder.senses_mapping}\n'
            )
            # encoder.senses_mapping = ckpt['hyper_parameters']['senses']
        else:
            ckpt_path = f'experiments/{exp_name.replace(args.experiment, args.restart_from)}/best.ckpt'

        model = model_class.load_from_checkpoint(
            ckpt_path,
            strict=False,
            encoder=encoder,
            lr=args.lr
        )
    else:
        model = model_class(encoder,
                            lr=args.lr,
                            warmup_init_lr=args.warmup_init_lr,
                            min_lr=args.min_lr,
                            warmup_steps=args.warmup_steps,
                            label_smoothing=args.smoothing,
                            projection_lr=args.projection_lr,
                            pretrained=args.pretrained)

    system = TranslationSystem(model, encoder)

    evaluation_callback = FullEvaluationCallback.from_datamodule(system, module, args.warmup_steps)

    logger = True  # default logger value for Trainer

    if not args.offline:
        logger = WandbLogger(
            name=exp_name,
            save_dir=folder,
            offline=True,
            project="mt-wsd",
            entity="mt-wsd",
            log_model=False,
            tags=tags,
            config=args,
        )

    sw_counter = SubwordsMonitor(pad_token_id=encoder.tokenizer.pad_token_id)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    monitor = args.monitor
    monitor_mode = args.monitor_mode

    checkpoint_callback = LinkBestModelCheckpoint(
        monitor=monitor,
        dirpath=folder,
        filename='{epoch:02d}-{' + monitor + ':.5f}',
        save_top_k=args.save_top,
        mode=monitor_mode,
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=args.patience,
        mode=monitor_mode
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[evaluation_callback, early_stopping, lr_monitor, sw_counter, checkpoint_callback],
        logger=logger,
        resume_from_checkpoint=resume_from,
        # check_val_every_n_epoch=25,
    )
    trainer.fit(model, datamodule=module)
    # trainer.test(model, test_dataloaders=test_loader)


if __name__ == '__main__':
    main()
