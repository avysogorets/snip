import tensorflow as tf
from tensorflow.keras import layers,models

class Lenet300100():
    def __init__(self,shape,initializer,decay=0.,batchnorm=False):
        self.shape=shape
        self.decay=decay
        self.epsilon=0.00000001
        self.initializer=initializer
        self.batchnorm=batchnorm
    def build(self):
        inputs=layers.Input(shape=self.shape)
        x=layers.Flatten()(inputs)
        x=layers.Dense(units=300,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_dense1')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Dense(units=100,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_dense2')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Dense(units=10,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_dense3')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('softmax')(x)
        model=models.Model(inputs,x)
        weight_tensors=[]
        for layer in range(len(model.layers)):
            if  'tensor' in model.layers[layer].name:
                weight_tensors.append(layer)
        return model,weight_tensors

class Conv6():
    def __init__(self,shape,decay,initializer,batchnorm=False):
        self.shape=shape
        self.decay=decay
        self.epsilon=0.00001
        self.initializer=initializer
        self.batchnorm=batchnorm
    def build(self):
        inputs=layers.Input(shape=self.shape)
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_conv1')(inputs)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_conv2')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_conv3')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_conv4')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_conv5')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",strides=(1,1),activation=None,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_conv6')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')(x)
        x=layers.Flatten()(x)
        x=layers.Dense(units=256,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_dense1')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Dense(units=256,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_dense2')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('relu')(x)
        x=layers.Dense(units=10,use_bias=True,bias_initializer='zeros',kernel_initializer=self.initializer,activation=None,kernel_regularizer=tf.keras.regularizers.l2(self.decay),name='tensor_dense3')(x)
        if self.batchnorm:
            x=layers.BatchNormalization(epsilon=self.epsilon)(inputs=x,training=True)
        x=layers.Activation('softmax')(x)
        model=models.Model(inputs,x)
        weight_tensors=[]
        for layer in range(len(model.layers)):
            if  'tensor' in model.layers[layer].name:
                weight_tensors.append(layer)
        return model,weight_tensors