import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from DataHelper import ImagesDataset
from torch.autograd import Variable

img_shape = (150, 150)

custom_transform = transforms.Compose([
                transforms.Resize(img_shape[0] + 20),
                transforms.RandomCrop(img_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

class DeepAutoencoder(nn.Module): 
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
         # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(size=img_shape, mode="bilinear", align_corners=True)

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        #out = self.relu(self.conv4(out))
        return out

    def decode(self, z):
        out = self.relu(self.deconv1(z))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        #out = self.relu(self.deconv4(out))
        return self.upsample(out)

    def e1(self, x):
        return self.encode(x)

    def e2(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        #out = self.relu(self.conv3(out))
        return out
    
    def e3(self, x):
        out = self.relu(self.conv1(x))
        #out = self.relu(self.conv2(out))
        return out
    '''
    def e4(self, x):
        out = self.relu(self.conv1(x))
        return out
    '''
    def d1(self, x):
        return self.decode(x)

    def d2(self, x):
        out = self.relu(self.deconv2(x))
        out = self.relu(self.deconv3(out))
        #out = self.relu(self.deconv4(out))
        return self.upsample(out)

    def d3(self, x):
        out = self.relu(self.deconv3(x))
        #out = self.relu(self.deconv4(out))
        return self.upsample(out)
    
    def forward(self, x) :
        return self.decode(self.encode(x))

'''
    def d4(self, x):
        out = self.relu(self.deconv4(x))
        return self.upsample(out)'''

    
def MSE(x, target) :
    return torch.sum(torch.pow((x - target),2)) / x.data.nelement()

if __name__ == "__main__" :

    num_epochs = 1000
    batch_size = 20
    nchannels = 3
    alpha = 1

    # Training the autoencoder on COCO Dataset 
    transformed_data = ImagesDataset(root_dir="data/val2017/", transform=custom_transform)

    dataloader = DataLoader(transformed_data, shuffle=True, batch_size=batch_size, num_workers=2)

    AutoE = DeepAutoencoder()
    #AutoE = DeepAutoencoder().cuda()
        
    try:
        AutoE = torch.load("models/autoencoder.pt")
        #pass
    except:
        raise "Can't load Autoencoder!"
    
    try:
        dummy = torch.arange(1*3*150*150).reshape(1,3,150,150) / (1*3*150*150)
        a = AutoE.d1(AutoE.e1(dummy))
        a = AutoE.d2(AutoE.e2(dummy))
        a = AutoE.d3(AutoE.e3(dummy))
        #a = AutoE.d4(AutoE.e4(dummy))
        del a
        del dummy
    except:
        raise "Some shit happened......"
    
    optimizer = torch.optim.Adam(AutoE.parameters(), lr=0.0001)
    
    itr = 0
    checkpoint = 10
    for epoch in range(num_epochs) :
        for data in dataloader :
            
            img = Variable(data)
            #img.cuda()

            optimizer.zero_grad()

            output = AutoE(img)
            loss = MSE(output, img) #+ alpha * MSE(AutoE.encode(output), AutoE.encode(img))
            #loss = MSE(AutoE.encode(output), AutoE.encode(img))
            
            loss.backward()
            optimizer.step()
            
            print('epoch [{}/{}], loss:{:.4f}, uIter: {}'.format(epoch+1, num_epochs, loss.item(), itr))
            itr += 1
            if itr % checkpoint == 0 :
                torch.save(AutoE, 'models/autoencoder.pt')
    torch.save(AutoE, 'models/autoencoder.pt')