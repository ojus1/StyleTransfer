from utils import Whiten, Blend
from utils_source import wct_np, wct_tf
from keras.models import load_model, Model
from keras.layers import UpSampling2D, MaxPool2D, Conv2D, Conv2DTranspose, Input, ZeroPadding2D
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import keras.backend as K
from PIL import Image
import numpy as np
from glob import glob



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
'''    
input_layer = Input(shape=(img_shape[0],img_shape[1],3))

l = Conv2D(3, (3, 3), strides=1, activation='relu', padding='same')(input_layer) #1
l = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(l)	#2
l = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(l)	#3
l = MaxPool2D(pool_size=(3,3), strides=1, padding='same')(l) #4
l = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(l) #5
l = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(l) #6
l = MaxPool2D(pool_size=(3,3), strides=1, padding='same')(l) #7
l = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(l) #8
l = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same')(l) #9
l = MaxPool2D(pool_size=(3,3), strides=1, padding='same')(l) #10
l = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(l) #11
l = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(l) #12

l = Conv2DTranspose(256, (3, 3), strides=1, padding='same')(l) #14
l = Conv2DTranspose(256, (3, 3), strides=1, padding='same')(l) #15
l = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(l) #16
l = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(l) #17
l = Conv2DTranspose(64, (3, 3), strides=1, padding='same')(l) #18
l = Conv2DTranspose(64, (3, 3), strides=1, padding='same')(l) #19
l = Conv2DTranspose(32, (3, 3), strides=1, padding='same')(l) #20
l = Conv2DTranspose(32, (3, 3), strides=1, padding='same')(l) #21
l = Conv2DTranspose(3, (3, 3), strides=1, padding='same')(l) #22
'''
#autoencoder = Model(inputs=input_layer, outputs=l)
autoencoder = load_model("production_models/autoencoderLvl2.hdf5")
'''
try:
    autoencoder.load_weights("models/sample_weights_xtradeep.hdf5")
except:
    print("Can't Load weights")
    pass
'''
autoencoder.summary()

def e1(x):
    temp = K.function([autoencoder.layers[0].input],[autoencoder.layers[1].output])
    #return ZeroPadding2D(padding=(12, 12))(temp([x])[0])
    return temp([x])[0]

def d1(x):
    temp = K.function([autoencoder.layers[2].input],[autoencoder.layers[2].output])
    return temp([x])[0]

def e2(x):
    temp = K.function([autoencoder.layers[0].input],[autoencoder.layers[3].output])
    #return ZeroPadding2D(padding=(12, 12))(temp([x])[0])
    return temp([x])[0]

def d2(x):
    temp = K.function([autoencoder.layers[4].input],[autoencoder.layers[4].output])
    return temp([x])[0]

def e3(x):
    temp = K.function([autoencoder.layers[0].input],[autoencoder.layers[3].output])
    #return ZeroPadding2D(padding=(12, 12))(temp([x])[0])
    return temp([x])[0]

def d3(x):
    temp = K.function([autoencoder.layers[4].input],[autoencoder.layers[4].output])
    return temp([x])[0]

def Stylize(content, style, layer_list, alpha=0.6):
    query = sorted(layer_list)
    #query.reverse()
    query = list(map(str, query))

    temp_content = content
    for item in query:
        temp_cF = eval('e'+item+'(temp_content)')
        temp_sF = eval('e'+item+'(style)')
        #temp_csF = Blend(temp_cF, temp_sF, alpha)
        temp_csF = wct_tf(temp_cF, temp_sF, alpha)
        temp_content = eval('d'+item+'(temp_csF)')

    return temp_content

content_paths = glob('sample/content_imgs/original/*')
content = LoadImg(content_paths[3], img_shape)

style_paths = glob('sample/style_imgs/original/*')
style = LoadImg(style_paths[1], img_shape)

target = Stylize(content, style, [2], 0.8).squeeze(0)
UnloadImg(target, 'sample/stylized/1.jpeg')
