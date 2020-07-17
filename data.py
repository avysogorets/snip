import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets,layers,models

def cifar10():
  (train_X,train_Y),(test_X,test_Y)=datasets.cifar10.load_data()
  train_images,test_images=train_X/255.0,test_X/255.0
  train_labels=np.zeros((len(train_Y),10))
  for j in range(len(train_Y)): 
    train_labels[j,train_Y[j]]=1
  test_labels=np.zeros((len(test_Y),10))
  for j in range(len(test_Y)): 
    test_labels[j,test_Y[j]]=1
  return train_images,train_labels,test_images,test_labels

def mnist():
  (train_X,train_Y),(test_X,test_Y)=datasets.mnist.load_data()
  train_images,test_images=train_X/255.0,test_X/255.0
  train_labels=np.zeros((len(train_Y),10))
  for j in range(len(train_Y)): 
    train_labels[j,train_Y[j]]=1
  test_labels=np.zeros((len(test_Y),10))
  for j in range(len(test_Y)): 
    test_labels[j,test_Y[j]]=1
  return train_images,train_labels,test_images,test_labels