from math import exp
from typing_extensions import OrderedDict
import yaml
import argparse
import numpy as np

from models import *
from experiment import Pix2pixExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger


#read/set default config file
parser = argparse.ArgumentParser(description='Generic runner for pix2pix models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/pix2pix_terrain.yaml')
parser.add_argument('--pretrained_model', '-m', dest='model_file',
                    help="path to pretrained model file to start with")

#load config file
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

#initialise logger
tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False


gen_model = pix2pix_model[config['model_params']['gen_name']](config['exp_params']['in_channels'],config['exp_params']['out_channels'])
disc_model = pix2pix_model[config['model_params']['disc_name']](config['exp_params']['in_channels'])

if args.model_file:
    experiment = Pix2pixExperiment.load_from_checkpoint(args.model_file, gen_model=gen_model,disc_model=disc_model,params=config['exp_params'])
    print("[INFO] Loaded pretrained model")
else:

    experiment = Pix2pixExperiment(gen_model,disc_model,config['exp_params'])
    print("[INFO] Loaded randomly initialized model")


runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 min_epochs=1,
                 logger=tt_logger,
                 flush_logs_every_n_steps=100,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=5,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)