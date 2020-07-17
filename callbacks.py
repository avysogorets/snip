import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import prune

def scheduler(epoch,lr):
  if epoch%25==0 and epoch>0:
    return lr*(1./1.1)
  else:
    return lr

class SubnetworkCallback(tf.keras.callbacks.Callback):
  def __init__(self,model,tensors,masks):
    super().__init__()
    self.model=model
    self.tensors=tensors
    self.masks=masks
  def on_train_batch_begin(self,batch,logs={}):
    for tensor in self.tensors:
      curr_b=self.model.layers[tensor].get_weights()[1]
      curr_w=self.model.layers[tensor].get_weights()[0]
      self.model.layers[tensor].set_weights((curr_w*self.masks[self.tensors.index(tensor)],curr_b))

class InfoCallback(tf.keras.callbacks.Callback):
  def __init__(self,model,tensors,interval,test):
    super().__init__()
    self.model=model
    self.test_X,self.test_y=test
    self.tensors=tensors
    self.interval=interval
    self.iteration=0
    self.epoch=1
  def on_train_batch_begin(self,batch,logs={}):
    if self.iteration%self.interval==0:
      test_metrics=self.model.evaluate(x=self.test_X,y=self.test_y,batch_size=len(self.test_y),verbose=False)
      print("[iteration/epoch: {}/{}][sparsity: {}][val loss: {:.4f}][val acc: {:.4f}]".format(self.iteration,self.epoch,prune.sparsity([self.model.layers[layer].get_weights()[0] for layer in self.tensors]),test_metrics[0],test_metrics[1]))
    self.iteration+=1
  def on_epoch_end(self,batch,logs={}):
    self.epoch+=1
