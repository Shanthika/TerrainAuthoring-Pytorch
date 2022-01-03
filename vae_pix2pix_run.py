import yaml
import argparse
import numpy as np
import cv2
from models import *
from experiments.vae_experiment import VAEXperiment
from experiments.pix2pix_experiment import Pix2pixExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader 
from terrain_loader import TerrainDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import callbacks


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
                    


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


#Vae Model
vae_model = vae_models[config['vae_model_params']['name']](**config['vae_model_params'])

# pix2pix model
gen_model = pix2pix_model[config['pix2pix_model_params']['gen_name']](config['exp_params']['in_channels'],config['exp_params']['out_channels'])
disc_model = pix2pix_model[config['pix2pix_model_params']['disc_name']](config['exp_params']['in_channels'])


if config['vae_model_params']['load_model'] and config['pix2pix_model_params']['load_model']:
    experiment_vae = VAEXperiment.load_from_checkpoint(config['vae_model_params']['pretrained_model'], vae_model=vae_model,params=config['exp_params'])
    experiment_p2p = Pix2pixExperiment.load_from_checkpoint(config['pix2pix_model_params']['pretrained_model'], gen_model=gen_model,disc_model=disc_model,params=config['exp_params'])
    print("[INFO] Loaded pretrained model")
else:
    experiment = VAEXperiment(vae_model, config['exp_params'])
    print("[INFO] Loaded randomly initialized model")


checkpoint_callback = ModelCheckpoint(
	monitor='val_loss',
	save_last=True,
	save_top_k=3
)


if config['exp_params']['train']:

    print(f"======= Training {config['model_params']['name']} =======")
    # runner.fit(experiment)
    # runner.save_checkpoint()

else:
    print(f"======= Testing =======")
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
        # minv, maxv = torch.min(result), torch.max(result)
        new = (result+1)*127.5
        return torch.squeeze(new).detach().numpy().transpose((1,2,0)).astype(np.uint8)

    vae_model.eval()
    gen_model.eval()

    for ip, op in sample_dataloader:
        res = vae_model(ip)[0]

        vae_res = torch.squeeze(res*255).detach().numpy().transpose((1,2,0)).astype(np.uint8)
        res = res*2-1

        res = gen_model(res)
        # print(len(res))
        res = denormalize(res)
        op = denormalize(op)

        cv2.imshow('sampled',vae_res)
        cv2.imshow('generated',res)
        cv2.imshow('desired',op)
        cv2.imwrite('sampled.png',vae_res)
        cv2.imwrite('generated.png',res)
        cv2.imwrite('desired.png',op)
        cv2.waitKey()