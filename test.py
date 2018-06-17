import torch
import torchvision
from torchvision import transforms
from PIL import Image
from DataHelper import imshow, ImagesDataset, trnsf, img_shape
from Autoencoder import Autoencoder
from Reconstructor import Reconstructor
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_transform = transforms.Compose([
                transforms.Resize(img_shape[0] + 20),
                transforms.RandomCrop((img_shape[0], img_shape[1])),
                transforms.ToTensor(),
                #transforms.Normalize(mean=mean, std=std),
])

img_path = "sample/content_img/Image-1.jpg"
img = Image.open(img_path)
img = custom_transform(img)

plt.figure()
imshow(img)
plt.show()