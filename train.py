import argparse
import collections
import torch
import torch.multiprocessing
import numpy as np
from parse_config import ConfigParser
from trainers import *
import subprocess


# fix random seeds for reproducibility
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#torch.use_deterministic_algorithms(True)

DEFAULT_SEED = 123

def main(config):
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')

    # init logger
    logger = config.get_logger('train')

    # log the random seed
    seed = config["seed"] if config["seed"] is not None else DEFAULT_SEED
    logger.info("Random seed: {}".format(seed))

    # log the current git hash
    if config["git_hash"] is not None:
        hash = config["git_hash"]
    else:
        hash = subprocess.check_output(["git", "describe", "--always"]).strip()
    logger.info("Git hash: {}".format(hash))

    # print training session description to logs
    logger.info("Description: {}".format(config["description"]))

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer_class = globals()[config['trainer']['type']]
    trainer = trainer_class(config)
    if config['eval']:
        trainer.eval(config['eval'])
    else:
        trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--dry_run', default=False, type=bool,
                      help='If true, disables logging of models to disk and tags to git (default: False)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of this training session')
    args.add_argument('-m', '--message', default=None, type=str,
                      help='description of this training session')
    args.add_argument('-g', '--git_hash', default=None, type=str,
                      help='manually enter git hash in case it\'s not available locally (e.g. remote execution)')
    args.add_argument('-e', '--eval', default=None, type=str, help='evaluate on the "train", "valid" or "test" sets')
    args.add_argument('-v', '--vis', default=False, action='store_true', help='visualize evaluation')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--ld', '--log_dir'], type=str, target='trainer;save_dir')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
