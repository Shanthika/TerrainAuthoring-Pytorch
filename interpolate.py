 
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
                        batch_size= 10,
                        num_workers=config['exp_params']['n_workers'],
                        shuffle = True,
                        drop_last=False)

def denormalize(result):
    new = (result+1)*127.5
    return torch.squeeze(new).detach().numpy().transpose((1,2,0)).astype(np.uint8)

vae_model.eval()
gen_model.eval()

for ip, op,_ in sample_dataloader: 
    ip1 = torch.unsqueeze(ip[0], 0)
    ip2 = torch.unsqueeze(ip[1], 0)
    

    op1 = torch.unsqueeze(op[0], 0)
    op2 = torch.unsqueeze(op[1], 0)

    mu1,var1 = vae_model.encode(ip1)
    mu2,var2 = vae_model.encode(ip2)
    std1 = torch.exp(0.5 *  var1)
    eps1 = torch.randn_like(std1)
    z1 = eps1 * std1 + mu1

    std2 = torch.exp(0.5 *  var2)
    eps2 = torch.randn_like(std2)
    z2 = eps2 * std2 + mu2

    # for i in range(4):
    ratios = np.linspace(0, 1, num=10)
    vectors = []

    for ratio in ratios:
        
        v = (1.0 - ratio) * z1  + ratio * z2
        vectors.append(v)

    images = [] 

    for j,v in enumerate(vectors):
       logits = vae_model.decode(v)
       images.append(logits)

    for i,j in enumerate(images):
        res=j
        res = gen_model(res) 
        res = denormalize(res)
        res = cv2.GaussianBlur(res, (7,7), 0)

        cv2.imwrite("images/interpolation/int_"+str(i+1)+'.png',res) 
        

    break 
