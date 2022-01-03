from pytorch_lightning import callbacks
import yaml
import argparse
import numpy as np
import cv2
from models import *
from experiments.vae_experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader 
from terrain_loader import TerrainDataset
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
                    
parser.add_argument('--pretrained_model', '-m', dest='model_file',
                    help="path to pretrained model file to start with")

 

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


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

model = vae_models[config['model_params']['name']](**config['model_params'])

if args.model_file:
    experiment = VAEXperiment.load_from_checkpoint(args.model_file, vae_model=model,params=config['exp_params'])
    print("[INFO] Loaded pretrained model")
else:
    experiment = VAEXperiment(model, config['exp_params'])
    print("[INFO] Loaded randomly initialized model")

checkpoint_callback = ModelCheckpoint(
	monitor='val_loss',
	save_last=True,
	save_top_k=3
)

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 min_epochs=1,
                 logger=tt_logger,
                 flush_logs_every_n_steps=100,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=5,
                 callbacks=[checkpoint_callback],
                 **config['trainer_params'])



if config['exp_params']['train']:

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    # runner.save_checkpoint()

else:
    print(f"======= Testing {config['model_params']['name']} =======")
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
        minv, maxv = torch.min(result), torch.max(result)
        new = (result/maxv)*255.0
        return torch.squeeze(new).detach().numpy().transpose((1,2,0)).astype(np.uint8)

    model.eval()

    for ip, op in sample_dataloader:
        res = model(ip)[0]
        # print(len(res))
        res = denormalize(res)
        op = denormalize(ip)

        cv2.imshow('in',res)
        cv2.imshow('out',op)
        cv2.imwrite('input.png',res)
        cv2.imwrite('output.png',op)
        cv2.waitKey()