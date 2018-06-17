import torch
import torchvision
from torchvision import transforms
from PIL import Image
from ShallowAutoencoder import ShallowAutoencoder
from torch.utils.data import DataLoader
from DataHelper import imshow, ImagesDataset, trnsf
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_shape = (64, 64)

custom_transform = transforms.Compose([
                transforms.Resize(img_shape[0] + 20),
                transforms.RandomCrop(img_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transformed_data = ImagesDataset(root_dir="sample/content_img/", transform=custom_transform)
dataloader = DataLoader(transformed_data, shuffle=True, batch_size=1, num_workers=1)

AutoE = ShallowAutoencoder()
    
try :    
    AutoE = torch.load("production_models/autoencoder.pt")
    print("Loaded model successfully.")
except :
    raise "Can't load models!"

unloader = transforms.ToPILImage()
    
i = 1
for data in dataloader :
    print(data.shape)
    img = Variable(data)
    
    unloader(img.view(3, img_shape[0], img_shape[1])).save("sample/autoencoder_out/transformed/" + str(i) +"transformed.jpg")

    out = AutoE.forward(img).view(3, img_shape[0], img_shape[1])

    out_img = unloader(out)
    out_img.save("sample/autoencoder_out/" + str(i) + ".jpg")
    i += 1
 