import keras.backend as K
from keras.models import load_model, Model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import numpy as np
import tensorflow as tf
#from io import StringIO

class Stylize():
    def __init__(self):
        self.levels = {1:"production_models/autoencoderLvl1.hdf5",
                       2:"production_models/autoencoderLvl2.hdf5",
                       3:"production_models/autoencoderLvl3.hdf5"}
        self.in_out_pairs = {1:{"in":(0,1), "out":(2,2)},
                             2:{"in":(0,3), "out":(4,4)},
                             3:{"in":(0,4), "out":(5,7)},}
        self.img_shape = (256, 256)
        self.batch_size = 1
        
    def StyleTransfer(self, content, style, level, alpha=0.6):
        self.content = self.LoadImg(content)
        self.style = self.LoadImg(style)
        
        model = load_model(self.levels[level])
        e = K.function([model.layers[self.in_out_pairs[level]["in"][0]].input],
                                        [model.layers[self.in_out_pairs[level]["in"][1]].output])
        d = K.function([model.layers[self.in_out_pairs[level]["out"][0]].input],
                                        [model.layers[self.in_out_pairs[level]["out"][1]].output])
        
        temp_content = self.content
        temp_cF = e([temp_content])[0]
        temp_sF = e([self.style])[0]
        temp_csF = self._wct_tf(temp_cF, temp_sF, alpha)
        temp_content = d([temp_csF])[0]

        del model
        del e
        del d
        return temp_content
    
    def LoadImg(self, filepath):
        img = Image.open(filepath)
        img = img.resize(self.img_shape)
        img = img_to_array(img)
        img = img / 255.
        img = np.expand_dims(img, 0)
        return img

    def UnloadImg(self, img, filepath):
        try:
            out = np.array(img).squeeze(0)
        except:
            out = np.array(img)
        out = array_to_img(out)
        out.save(filepath)
    
    #Borrowed from : https://github.com/eridgd/WCT-TF/blob/master/ops.py "Borrowed" :P 
    def _wct_tf(self, Ic, Is, alpha, eps=1e-8):
        '''TensorFlow version of Whiten-Color Transform
        Assume that content/style encodings have shape 1xHxWxC
        See p.4 of the Universal Style Transfer paper for corresponding equations:
        https://arxiv.org/pdf/1705.08086.pdf
        '''
        # Remove batch dim and reorder to CxHxW
        content_t = tf.transpose(tf.squeeze(Ic), (2, 0, 1))
        style_t = tf.transpose(tf.squeeze(Is), (2, 0, 1))

        Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
        Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

        # CxHxW -> CxH*W
        content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
        style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

        # Content covariance
        mc = tf.reduce_mean(content_flat, axis=1, keepdims=True)
        fc = content_flat - mc
        fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

        # Style covariance
        ms = tf.reduce_mean(style_flat, axis=1, keepdims=True)
        fs = style_flat - ms
        fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

        # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
        with tf.device('/cpu:0'):  
            Sc, Uc, _ = tf.svd(fcfc)
            Ss, Us, _ = tf.svd(fsfs)

        ## Uncomment to perform SVD for content/style with np in one call
        ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
        # @jit
        # def np_svd(content, style):
        #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
        #     Uc, Sc, _ = np.linalg.svd(content)
        #     Us, Ss, _ = np.linalg.svd(style)
        #     return Uc, Sc, Us, Ss
        # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

        # Filter small singular values
        k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
        k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

        # Whiten content feature
        Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
        fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

        # Color content with style
        Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
        fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fc_hat)

        # Re-center with mean of style
        fcs_hat = fcs_hat + ms

        # Blend whiten-colored feature with original content feature
        blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

        # CxH*W -> CxHxW
        blended = tf.reshape(blended, (Cc,Hc,Wc))
        # CxHxW -> 1xHxWxC
        blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

        return blended