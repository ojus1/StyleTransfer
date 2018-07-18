from StyleTransfer import Stylize
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import keras.backend as K
from PIL import Image
from glob import glob
import numpy as np

img_shape = (256, 256)
batch_size = 1

def LoadImg(filepath, img_shape):
    img = Image.open(filepath)
    img = img.resize(img_shape)
    img =  img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, 0)
    return img

def UnloadImg(img, filepath):
    try:
        out = np.array(img).squeeze(0)
    except:
        out = np.array(img)
    out = array_to_img(out)
    out.save(filepath)

content_paths = glob('sample/content_imgs/original/*')

style_paths = glob('sample/style_imgs/original/*')

stylize = Stylize()

itr = 1
for i in range(len(content_paths)):
    for j in range(len(style_paths)):
        for level in range(1,4):
            target = stylize.StyleTransfer(content_paths[i], style_paths[j], level, np.random.uniform(0., 1.)).squeeze(0)
            UnloadImg(target, 'sample/stylized/'+str(itr)+'.jpeg')
            itr += 1