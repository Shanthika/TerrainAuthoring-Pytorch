from pytorch_lightning import callbacks
import yaml
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
from models import *
from experiments.vae_experiment import VAEXperiment
from experiments.vae_pix2pix_exp import Pix2pixExperiment
from torch.utils.data import DataLoader 
from terrain_loader import TerrainDataset




def denormalize(result):
        new = (result+1)*127.5
        return torch.squeeze(new).detach().numpy().transpose((1,2,0)).astype(np.uint8)


def display(ip, op, val=0):
    res = vae_model(ip)[0] 
    
    res = gen_model(res) 
    res = denormalize(res)
    ip = denormalize(ip)
    
    # res = cv2.GaussianBlur(res, (7, 7), 0)
    plt.figure(figsize=(10,10))

    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.gist_earth

    # plt.imshow(ls.hillshade(res,vert_exag=1 ), cmap='gray')
    # rgb = ls.shade(res, cmap=cmap, blend_mode='overlay' ,vert_exag=1)
    # plt.savefig('samples/render.png')
    print(ip.shape)
    plt.imshow(ls.hillshade(res[:,:,0],vert_exag=1,dy=10,dx=10), cmap='gray')
    rgb = ls.shade(res[:,:,0], cmap=cmap, blend_mode='overlay' ,vert_exag=1)
    plt.imshow(rgb)
    plt.axis("off")
    cv2.imwrite("ui_study/inp.png",ip)
    cv2.imwrite("ui_study/dem.png",res)


    plt.show()



with open("configs/test.yml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(".\n\n",exc)

#Vae Model
vae_model = vae_models[config['vae_model_params']['name']](**config['vae_model_params'])

# pix2pix model
gen_model = pix2pix_model[config['pix2pix_model_params']['gen_name']](config['exp_params']['in_channels'],config['exp_params']['out_channels'])
disc_model = pix2pix_model[config['pix2pix_model_params']['disc_name']](config['exp_params']['in_channels'])


if config['vae_model_params']['load_model'] :
    experiment_p2p = Pix2pixExperiment.load_from_checkpoint(config['pix2pix_model_params']['pretrained_model'], gen_model=gen_model,disc_model=disc_model,vae_model=vae_model,params=config['exp_params'])
    experiment_vae = VAEXperiment.load_from_checkpoint(config['vae_model_params']['pretrained_model'], vae_model=vae_model,params=config['exp_params'])
    print("[INFO] Loaded pretrained model")


vae_model.eval()
gen_model.eval()

if __name__=='__main__':
    dataset = TerrainDataset(root = "./",#config['exp_params']['data_path'],
                            train=False,
                            hide_green=config['exp_params']['hide_green'],
                            norm=config['exp_params']['norm'])

    sample_DataLoader = DataLoader(dataset,
                            batch_size= 1,
                            num_workers=config['exp_params']['n_workers'],
                            shuffle = True,
                            drop_last=False)

    for ip, op in sample_DataLoader:
        display(ip,op)
        
        break