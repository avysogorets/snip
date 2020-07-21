import tensorflow as tf
import numpy as np
import initializers
import callbacks
import prune
import data
import os
import models
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='mnist',help='dataset to use')
parser.add_argument('--sample',type=str,default='0',help='sample')
parser.add_argument('--save',type=int,default=1,help='sample')
parser.add_argument('--architecture',type=str,default='lenet300100',help='network architecture to use')
parser.add_argument('--pruner',type=str,default='snip_global',help='pruner of choice')
parser.add_argument('--target_sparsity',type=float,default=0.9,help='level of sparsity to achieve')
parser.add_argument('--batch_size_train',type=int,default=100,help='number of examples per mini-batch for training')
parser.add_argument('--batch_size_snip',type=int,default=10000,help='number of examples per mini-batch for snipping')
parser.add_argument('--iterations',type=int,default=100000,help='number of training iterations')
parser.add_argument('--momentum',type=float,default=0.9,help='sgd momentum')
parser.add_argument('--initializer',type=str,default='vs',help='initializer of choice')
parser.add_argument('--batchnorm',type=int,default=1)
parser.add_argument('--interval',type=int,default=100,help='check interval during training')
parser.add_argument('--weight_decay',type=float,default=0.0005,help='L2 weight regularization constant')
parser.add_argument('--lr',type=float,default=1e-1,help='initial learning rate')
args=parser.parse_args()

save_prefix='results/'+args.architecture+'/'+args.pruner+'/'+args.sample+'_'
train_X,train_y,test_X,test_y=data.get_data(args.data)
initializer=initializers.get_initializer(args.initializer)
model,tensors=models.get_model(train_X[0].shape,args.architecture,initializer,args.batchnorm,args.weight_decay)
pruner=prune.Pruner(args.pruner)
epochs=int(args.batch_size_train*args.iterations/len(train_X))
snip_indices=np.random.choice(range(len(train_X)),replace=False,size=args.batch_size_snip)
masks=pruner.prune(model,tensors,args.target_sparsity,batch_X=train_X[snip_indices],batch_y=train_y[snip_indices])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr,momentum=args.momentum),loss='categorical_crossentropy',metrics=['accuracy'])
callbacks=[callbacks.SubnetworkCallback(model,tensors,masks),callbacks.InfoCallback(model,tensors,args.interval,(test_X,test_y),save_prefix,args.save),tf.keras.callbacks.LearningRateScheduler(callbacks.scheduler)]
model.fit(x=train_X,y=train_y,epochs=epochs,shuffle=True,verbose=False,batch_size=args.batch_size_train,validation_data=(test_X,test_y),callbacks=callbacks)
if args.save:
  np.save(save_prefix+'mask.npy',masks[0])