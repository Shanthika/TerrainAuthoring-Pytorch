# TerrainAuthoring-Pytorch

This is the code for the paper **Deep Generative Framework for Interactive 3D Terrain Authoring and Manipulation**(link).

We refered the [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) repository for the VAE implementation.


## Installation.
Please install [conda](https://docs.anaconda.com/anaconda/install/index.html). Create a new environment and install all the dependencies using the following command
```
conda env create --file environment.yml
```

## Dataset

## Experiments
 
## Training the model
To train the model in trained in two steps. To train the VAE model, set the *load_model* parameter in *train.yml* to false. To train the *pix2pix* model use the following command. The checkpoints will be saved in *logs* directory.
```
python train.py --config configs/train.yml
```


## Testing the model

## Use the UI

## Terrain Interpolation
The model can be used to smoothly interpolate between the given two terrains
![](./images/interpolation.gif)


## Terrain Varitions
