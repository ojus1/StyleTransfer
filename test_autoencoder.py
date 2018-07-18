#import plaidml.keras
#plaidml.keras.install_backend()
from keras.models import load_model, Model
from keras.layers import UpSampling2D, MaxPool2D, Conv2D, Conv2DTranspose, Input
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
import numpy as np
import glob



img_shape = (256, 256)
batch_size = 1

def LoadImg(filepath, img_shape):
    img = Image.open(filepath)
    img = img.resize(img_shape)
    img =  img_to_array(img)
    img = np.array(img, np.float32) / 255
    img = np.expand_dims(img, 0)
    return img

def UnloadImg(img, filepath):
    out = np.array(img).squeeze(0)
    out = array_to_img(out)
    out.save(filepath)
'''
input_layer = Input(shape=(200, 200, 3))

l = Conv2D(3, (3, 3), strides=1, activation='relu', padding='same')(input_layer)
l = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(l)
l = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(l)
l = MaxPool2D((2, 2), padding='same')(l)
l = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(l)
l = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(l)
l = MaxPool2D((2, 2), padding='same')(l)
l = Conv2D(200, (3, 3), strides=1, activation='relu', padding='same')(l)
l = Conv2D(200, (3, 3), strides=1, activation='relu', padding='same')(l)
l = MaxPool2D((2, 2), padding='same')(l)

l = Conv2DTranspose(200, (3, 3), strides=1, padding='same')(l)
l = Conv2DTranspose(200, (3, 3), strides=1, padding='same')(l)
l = UpSampling2D(size=2)(l)
l = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(l)
l = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(l)
l = UpSampling2D(size=2)(l)
l = Conv2DTranspose(64, (3, 3), strides=1, padding='same')(l)
l = Conv2DTranspose(64, (3, 3), strides=1, padding='same')(l)
l = Conv2DTranspose(3, (3, 3), strides=1, padding='same')(l)
l = UpSampling2D(size=2)(l)

'''

autoencoder = load_model("models/autoencoderLvl3.hdf5")
autoencoder.summary()
'''
try:
    autoencoder.load_weights("models/train_weightmaxpool.hdf5")
except:
    print("Can't Load weights")
    pass
'''
i = 1
for filepath in glob.glob('sample/content_imgs/original/*.jpeg'):
    print(i)
    img = LoadImg(filepath, img_shape)
    out = autoencoder.predict_on_batch(img)
    UnloadImg(out, 'sample/autoencoder_out/'+str(i)+'.jpeg')
    i += 1