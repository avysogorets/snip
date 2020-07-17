import tensorflow as tf
import numpy as np
from models import *
import initializers
import callbacks
import prune
import data
import sys
import argparse
import visualization

parser=argparse.ArgumentParser()

log=500
sparsity=0.8

train_X,train_y,test_X,test_y=data.mnist()
lenet=Lenet300100(train_X[0].shape,tf.keras.initializers.VarianceScaling,decay=0.0005)
model,tensors=lenet.build()
model.summary()
pruner=prune.Pruner('snip_layer')
model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
masks=pruner.prune(model,tensors,sparsity,batch_X=train_X[:100],batch_y=train_y[:100])
visualizer=visualization.Visualizer('mask')
visualizer.visualize(mask=masks[0])
callbacks=[callbacks.SubnetworkCallback(model,tensors,masks),callbacks.InfoCallback(model,tensors,log,(test_X,test_y)),tf.keras.callbacks.LearningRateScheduler(callbacks.scheduler)]
model.fit(x=train_X,y=train_y,epochs=50,shuffle=True,verbose=False,batch_size=100,validation_data=(test_X,test_y),callbacks=callbacks)