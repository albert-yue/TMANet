import argparse
import os
import os.path as osp
import sys
import socket
import time

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

# this_dir = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(this_dir, '..'))

from mmseg.apis import multi_gpu_test, single_gpu_test, set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import get_root_logger, get_dist_env


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--distributed', action='store_true', help='distributed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

# run() will be the training function
def run(rank, size, hostname, cfg, args, logger):
    print(f"{hostname}: {rank} of {size}")
    # See this : https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
    # your code will probably look like this -
    # model = DistributedDataParallel(model, device_ids=[rank%2], output_device=None)
    # Here device_ids=rank%2 because we launch two python processes per node.
    # (rank%2) assigns GPU 0 or 1 uniquely to each. This is the LOCAL rank of each process, whereas the
    # variable "rank" contains the GLOBAL rank which ranges from 0:(N-1). The LOCAL rank of each python
    # process is 0 or 1 and we use this to assign a unique GPU per process.
    # 
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=args.distributed,
        shuffle=False)

    print('Dataset length:', len(dataset), 'and DL:', len(data_loader))

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    if not args.distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[rank%2],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    if rank == 0:
        print('Done with test')
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, logger, **kwargs)


if __name__ == "__main__":
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    
    cfg = mmcv.Config.fromfile(args.config)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Dont change the following :
    global_rank, world_size = get_dist_env()
    
    hostname = socket.gethostname()

    # You have run dist.init_process_group to initialize the distributed environment
    # Always use NCCL as the backend. Gloo performance is pretty bad and MPI is currently
    # unsupported (for a number of reasons).     
    if args.distributed:
        dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

    # init the logger before other steps
    logger = None
    if args.eval:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'test_{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        if logger is not None:
            logger.info(f'Set random seed to {args.seed}, deterministic: '
                        f'{args.deterministic}')
        else:
            print(f'Set random seed to {args.seed}, deterministic: '
                  f'{args.deterministic}')

    # now run your distributed training code
    run(global_rank, world_size, hostname, cfg, args, logger)

