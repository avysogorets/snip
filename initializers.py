import tensorflow as tf
from  tensorflow.keras import backend as K
from tensorflow.python.framework import dtypes 
import numpy as np

def get_initializer(mode):
  if mode=='he_signed_const':
    return he_signed_const()
  elif mode=='he_normal' or mode=='kaiming_normal':
    return tf.keras.initializers.he_normal()
  elif mode=='glorot_normal':
    return tf.keras.initializers.GlorotNormal()
  elif mode=='variance_scaling' or mode=='vs':
    return tf.keras.initializers.VarianceScaling()

class he_signed_const(tf.keras.initializers.Initializer):
    def __init__(self,mult=1,pos_bias=0):
        self.pos_bias=pos_bias
        self.mult=mult
    def __call__(self,shape,dtype=dtypes.float32):
        if self.pos_bias<-0.5 or self.pos_bias>0.5:
            raise ValueError('he signed const init: pos_bias out of range')
        if len(shape)>2:
            sdev=np.sqrt(self.mult*1./(shape[0]*shape[1]*shape[3]))
        elif len(shape)==2:
            sdev=np.sqrt(self.mult*1./(shape[0]))
        signs=np.random.choice(a=(-1,1),size=shape,replace=True,p=(0.5+self.pos_bias,0.5-self.pos_bias))
        return K.variable(np.full(shape=shape,fill_value=sdev)*signs)