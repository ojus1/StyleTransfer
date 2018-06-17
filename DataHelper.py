import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image

img_shape = (224, 224)
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225] 

trnsf = transforms.Compose([
                transforms.Resize(img_shape[0] + 20),
                transforms.RandomCrop((img_shape[0], img_shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
])


class ImagesDataset(Dataset) :
    """Image Dataset"""
    def __init__(self, root_dir, transform=trnsf) :
        self.root_dir = root_dir
        self.transform = transform
        self.images_list = os.listdir(root_dir)
        self.images_list = [item for item in self.images_list if item.endswith('.jpg') or item.endswith('.bmp') or item.endswith('.jpeg')]
    
    def __len__(self) :
        return len(self.images_list)
    
    def __getitem__(self,idx) :
        img_name = self.images_list[idx]
        img = Image.open(self.root_dir + img_name).convert('RGB')

        if self.transform :
            img = self.transform(img)
        return img

def imshow(tensor, title=None, cuda=0) :
    unloader = transforms.ToPILImage()
    if cuda == 0 :
        image = tensor.cpu().clone()  
    else :
        image = tensor.cuda().clone()
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    #plt.show()

class Cifar10(Dataset) :
    def __init__(self) :
        self.cifar10_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=trnsf, download=True)
    def Data(self) :
        return self.cifar10_dataset