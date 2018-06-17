from ShallowAutoencoder import ShallowAutoencoder, custom_transform, img_shape
import torch 
import torchvision
from PIL import Image
from torch.autograd import Variable

def whiten_and_color(cF, sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1).double() + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF.double())

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)).double() ,(s_v[:,0:k_s].t()).double()), whiten_cF.double())
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature).double()
    return targetFeature

def transform(cF, sF, csF, alpha):
    cF = cF.double()
    sF = sF.double()
    C,W,H = cF.size(0),cF.size(1),cF.size(2)
    _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
    cFView = cF.view(C,-1)
    sFView = sF.view(C,-1)

    targetFeature = whiten_and_color(cFView, sFView)
    targetFeature = targetFeature.view_as(cF)
    ccsF = alpha * targetFeature + (1.0 - alpha) * cF
    ccsF = ccsF.float().unsqueeze(0)
    csF.data.resize_(ccsF.size()).copy_(ccsF)
    return csF

def StyleTransfer(Ic, Is, alpha=0.5) :
    try :
        model = torch.load("production_models/autoencoder.pt")
    except :
        raise "FATAL ERROR: CANT LOAD MODEL"
    
    if not (Image.isImageType(Ic) and Image.isImageType(Is)) :
        try :
            Ic = Image.open(Ic).convert('RGB')
            Is = Image.open(Is).convert('RGB')
        except :
            raise "Cannot load images."
    
    loader = custom_transform
    unloader = torchvision.transforms.ToPILImage()

    Ic = Variable(loader(Ic))
    Is = Variable(loader(Is))

    cF = model.encode(Ic.unsqueeze(0)).data.cpu().squeeze(0).view(64, 64*64)
    sF = model.encode(Is.unsqueeze(0)).data.cpu().squeeze(0).view(64, 64*64)
    #print(cF.shape)
    #print(sF.shape)
    csF = Variable(whiten_and_color(cF, sF))
    
    cF = cF.view(64, 64, 64).double()
    sF = sF.view(64, 64, 64).double()
    target = transform(cF, sF, csF, alpha)
    target = model.decode(target.float())

    target_img = unloader(target.view(3, img_shape[0], img_shape[1]))
    return target_img