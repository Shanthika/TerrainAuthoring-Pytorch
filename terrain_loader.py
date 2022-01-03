
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset 
import cv2


def show_images(f):
    cv2.imshow('a',f)
    cv2.waitKey(0)

class TerrainDataset(Dataset):

    def __init__(self, root, shuffle=True, transform=None, norm=1, train=False, hide_green=False):
        self.root = root
        if train:
            # with open(os.path.join(root, 'train.txt'), 'r') as f:
            #     self.files = f.read().split("\n")
            self.files = os.listdir(os.path.join(root, 'train'))
            self.image_dir = 'train'
        else:
            # with open(os.path.join(root, 'test.txt'), 'r') as f:
            #     self.files = f.read().split("\n")
            self.files = os.listdir(os.path.join(root, 'test'))
            self.image_dir = 'test'
        if shuffle:
            random.shuffle(self.files)
        
        self.transform = transform
        self.norm = norm
        self.train = train
        self.hide_green = hide_green

        self.__cache = {}
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        assert index<len(self), 'index out of range'
        
        if index in self.__cache:
            input_img, dem, file_path  = self.__cache[index]
        else:
            file_path = os.path.join(self.root, self.image_dir, self.files[index])
            orig = cv2.imread(file_path)
            input_img = orig[:,:256,:]
            dem = orig[:,256:,:]

            if self.hide_green:
                input_img[:,:,1] = 0
          
            input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
            dem = cv2.cvtColor(dem,cv2.COLOR_BGR2RGB)

            assert np.all(np.array(input_img.shape)==np.array(dem.shape)), "Shape mismatch"
            self.__cache[index] = (input_img, dem, file_path)

        # if self.train:
            # data augmentation
            # h,w,c = input_img.shape

            # input_img, dem = get_random_crop(input_img, dem, 256,256)
            # dem = get_random_crop(, 256,256)1
            
            # flip
            # if( torch.rand(()) > 0.5):
            #     input_img = cv2.flip(input_img, 1)
            #     dem = cv2.flip(dem, 1)
        
        if self.norm==0:
            #Normalize between [0,1]
            input_img = torch.from_numpy(input_img.transpose((2, 0, 1))).float().div(255)
            dem = torch.from_numpy(dem.transpose((2, 0, 1))).float().div(255)
        elif self.norm==1:
            #Normalize between [-1,1]
            input_img = torch.from_numpy(input_img.transpose((2, 0, 1))).float().div(127.5)-1
            dem = torch.from_numpy(dem.transpose((2, 0, 1))).float().div(127.5)-1
        elif self.norm==2:
            input_img = torch.from_numpy(input_img.transpose((2, 0, 1))).float().div(255)
            dem = torch.from_numpy(dem.transpose((2, 0, 1))).float().div(127.5)-1
        else:
            raise Exception("Invalid normalization scheme")
        
        if self.transform is not None:
            input_img = self.transform(input_img)
            dem = self.transform(dem)

        return input_img, dem, file_path


def get_random_crop(image, dem, crop_height, crop_width):

    image = cv2.resize(image, (286, 286), interpolation=cv2.INTER_NEAREST) # INTER_NEAREST, INTER_LINEAR
    dem = cv2.resize(dem, (286, 286), interpolation=cv2.INTER_NEAREST) # INTER_NEAREST, INTER_LINEAR

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]
    crop_dem = dem[y: y + crop_height, x: x + crop_width]

    return crop, crop_dem
