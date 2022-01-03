 
import yaml 
import numpy as np
import cv2
import argparse
from models import *
from experiments.vae_pix2pix_exp import Pix2pixExperiment 
from torch.utils.data import DataLoader 
from terrain_loader import TerrainDataset 



parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/test.yml')
                    
parser.add_argument('--var',
                    dest="var",
                    help =  'type of output',
                    default='single')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

#Vae Model
vae_model = vae_models[config['vae_model_params']['name']](**config['vae_model_params'])

# # pix2pix model
gen_model = pix2pix_model[config['pix2pix_model_params']['gen_name']](config['exp_params']['in_channels'],config['exp_params']['out_channels'])
disc_model = pix2pix_model[config['pix2pix_model_params']['disc_name']](config['exp_params']['in_channels'])

experiment_p2p = Pix2pixExperiment.load_from_checkpoint(config['pix2pix_model_params']['pretrained_model'], gen_model=gen_model,disc_model=disc_model,vae_model=vae_model,params=config['exp_params'])
print("Loaded pretrained model")


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
    new = (result+1)*127.5
    return torch.squeeze(new).detach().numpy().transpose((1,2,0)).astype(np.uint8)

vae_model.eval()
gen_model.eval()

if (args.var=='single'):
    for ip, op,_ in sample_dataloader:
        res = vae_model(ip)[0]

        vae_res = torch.squeeze(res*255).detach().numpy().transpose((1,2,0)).astype(np.uint8)
        res = gen_model(res)
        res = denormalize(res)
        op = denormalize(op)

        cv2.imwrite('images/sampled.png',vae_res)
        cv2.imwrite('images/generated.png',res)
        cv2.imwrite('images/desired.png',op)

        break

else:
    for ip, op,_ in sample_dataloader:
        mu,var = vae_model.encode(ip)

        for i in range(5):
            std = torch.exp(0.5 * var)
            eps = torch.randn_like(std)
            z = eps * 0.5 + mu
            res = vae_model.decode(z) 
            vae_res = torch.squeeze( res*255).detach().numpy().transpose((1,2,0)).astype(np.uint8)
        
        
            res = gen_model(res) 
            res = denormalize(res)
            cv2.imwrite("images/vae_"+str(i+1)+'.png',vae_res) 
            cv2.imwrite("images/dem_"+str(i+1)+'.png',res) 

        op = denormalize(op)
        cv2.imwrite('images/desired.png',op)

        break

