import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from DataHelper import ImagesDataset
from torch.autograd import Variable

img_shape = (64, 64)

custom_transform = transforms.Compose([
                transforms.Resize(img_shape[0] + 20),
                transforms.RandomCrop(img_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

class ShallowAutoencoder(nn.Module): 
    def __init__(self):
        super(ShallowAutoencoder, self).__init__()
         # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.upsample = nn.Upsample(size=img_shape, mode="bilinear")

    def encode(self, x):
        return self.relu(self.conv1(x)) 

    def decode(self, z):
        out = self.relu(self.deconv1(z))
        return self.upsample(out)

    def forward(self, x) :
        return self.decode(self.encode(x))

def MSE(x, target) :
    return torch.sum(torch.pow((x - target),2)) / x.data.nelement()

if __name__ == "__main__" :

    num_epochs = 1000
    batch_size = 128
    nchannels = 3
    alpha = 0.8
    img_shape = (64, 64)


    # Training the autoencoder on COCO Dataset 
    transformed_data = ImagesDataset(root_dir="data/val2017/", transform=custom_transform)

    dataloader = DataLoader(transformed_data, shuffle=True, batch_size=batch_size, num_workers=1)

    AutoE = ShallowAutoencoder()
    #AutoE = Autoencoder().cuda()
    try :
        AutoE = torch.load("models/autoencoder.pt")
    except :
        raise "Can't load Autoencoder!"

    optimizer = torch.optim.Adam(AutoE.parameters(), lr=1e-4)
    
    itr = 0
    checkpoint = 10
    for epoch in range(num_epochs) :
        for data in dataloader :
            
            img = Variable(data)
            #img.cuda()

            optimizer.zero_grad()

            output = AutoE(img)
            loss = MSE(output, img) + alpha * MSE(AutoE.encode(output), AutoE.encode(img))
            #loss = MSE(AutoE.encode(output), AutoE.encode(img))
            
            loss.backward()
            optimizer.step()
            
            print('epoch [{}/{}], loss:{:.4f}, uIter: {}'.format(epoch+1, num_epochs, loss.data[0], itr))
            itr += 1
            if itr % checkpoint == 0 :
                torch.save(AutoE, 'models/autoencoder.pt')
    torch.save(AutoE, 'models/autoencoder.pt')
