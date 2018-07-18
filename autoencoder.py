#import plaidml.keras
#plaidml.keras.install_backend()
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import UpSampling2D, MaxPool2D, Conv2D, Conv2DTranspose, Input, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np

img_shape = (256, 256, 3)
batch_size = 5

def pre_func(x):
    return (x - 0.5) 

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             rescale=1./255,
                             preprocessing_function=pre_func,
                             shear_range=0.2,
                             )

train_gen = datagen.flow_from_directory('../../Datasets/COCO/',
                                                   target_size=(img_shape[0], img_shape[1]),
                                                   class_mode=None,
                                                   batch_size=batch_size,
                                                   shuffle=True)

def fixed_generator(func):
    for batch in func:
        yield batch, batch

#TO DO Build Conv net using 2 layers of Convolutions on 512x512 size image

input_layer = Input(shape=img_shape)

l = Conv2D(3, (3, 3), strides=1, activation='relu', padding='same')(input_layer)
l = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(l)
l = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(l)
l = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same')(l)
l = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(l)
l = Conv2DTranspose(128, (3, 3), strides=1, padding='same')(l)
l = Conv2DTranspose(3, (3, 3), strides=1, padding='same')(l)

autoencoder = Model(inputs=input_layer, outputs=l)
autoencoder.summary()

try:
    #autoencoder.load_weights("models/train_weightmaxpool.hdf5")
    autoencoder = load_model("models/autoencoderLvl3.hdf5")
    pass
except:
    print("Can't Load weights")
    pass

autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc', 'mae'])

#autoencoder = load_model('models/autoencoderV3.hdf5')

checkpoint = ModelCheckpoint("models/autoencoderLvl3.hdf5", monitor='loss')
call_backs = [checkpoint]

autoencoder.fit_generator(fixed_generator(train_gen),
                          steps_per_epoch=10, 
                          epochs=1000,
                          verbose=1, 
                          shuffle=True,
                          workers=2, 
                          callbacks=call_backs)

'''
best_loss = 0
current_loss = 0

count = 1
itr = 0
checkpoint = 10
for Xbatch, Ybatch in fixed_generator(train_gen):
    
    autoencoder.train_on_batch(Xbatch, Ybatch)
    
    count += 1
    current_loss += autoencoder.test_on_batch(Xbatch, Ybatch)[0]
    
    itr += 1
    print("BackProp Iter: {}, loss: {:.4f}, best loss: {:4f}".format(itr, current_loss / count, best_loss))
    if itr % checkpoint == 0:
        count = 1
        current_loss = autoencoder.test_on_batch(Xbatch, Ybatch)
        if current_loss < best_loss:
            print("Saving model.....")
            best_loss = current_loss
            autoencoder.save("models/autoencoderv2.hdf5")
    
    elif itr == 0:
        best_loss = current_loss
'''