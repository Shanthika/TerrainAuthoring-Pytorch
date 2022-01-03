from math import exp
from typing_extensions import OrderedDict
import yaml
import argparse
import numpy as np
import cv2

from models import *
from pix2pix_experiment import Pix2pixExperiment
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
parser.add_argument('--pretrained_model', '-m', dest='model_file',required=True,
                    help="path to pretrained model file to start with")
parser.add_argument('--image_path', '-i', dest='img_path', required=True,
                    help="path to input for model")


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

# loading model
experiment = Pix2pixExperiment.load_from_checkpoint(args.model_file, gen_model=gen_model,disc_model=disc_model,params=config['exp_params'])
print("[INFO] Loaded pretrained model")


from torch.utils.data import DataLoader 
from terrain_loader import TerrainDataset

dataset = TerrainDataset(root = config['exp_params']['data_path'],
                        train=False,
                        hide_green=config['exp_params']['hide_green'],
                        norm=config['exp_params']['norm'])
sample_dataloader = DataLoader(dataset,
                                batch_size= 1,
                                num_workers=config['exp_params']['n_workers'],
                                shuffle = False,
                                drop_last=False)

def denormalize(result):
    new = (result+1)*127.5
    return torch.squeeze(new).detach().numpy().transpose((1,2,0)).astype(np.uint8)

gen_model.eval()

for ip, op in sample_dataloader:
    res = gen_model(ip)
    res = denormalize(res)
    op = denormalize(op)

    cv2.imshow('a',res)
    cv2.imshow('b',op)
    cv2.waitKey()

