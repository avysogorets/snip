import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def sparsity(matrices):
  s=[round(1-float(len(np.nonzero(matrix.reshape(-1))[0]))/np.prod(matrix.shape),4) for matrix in matrices]
  return tuple(s)

class Pruner(object):
  def __init__(self,mode):
    self.mode=mode
  
  def prune(self,model,tensors,sparsity,batch_X,batch_y):
    if self.mode=='snip_layer':
      return self.prune_snip_layer(model,tensors,sparsity,batch_X,batch_y)
    elif self.mode=='snip_global':
      return self.prune_snip_global(model,tensors,sparsity,batch_X,batch_y)
    elif self.mode=='random_global':
      return self.prune_random_global(model,tensors,sparsity)
    elif self.mode=='random_layer':
      if len(tensors)==3:
        return self.prune_random_layer(model,tensors,sparsity)
      else:
        raise RuntimeError('unknown pruning rates for this model type')
    elif self.mode=='base' or self.mode=='none':
      return [np.ones(model.layers[layer].get_weights()[0].shape) for layer in tensors]
    else:
      raise ValueError(f'unknown pruner "{self.mode}" encounterd.')

  def prune_snip_layer(self,model,tensors,sparsity,batch_X,batch_y):
    masks=[np.ones(model.layers[layer].get_weights()[0].shape) for layer in tensors]
    with tf.GradientTape(persistent=True) as tape:
      output=model(batch_X)
      loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(output,batch_y))
    abs_grads=[np.abs(tape.gradient(loss,model.layers[layer].trainable_weights)[0].numpy()) for layer in tensors]
    cs=[np.divide(abs_grads[tensors.index(layer)]*model.layers[layer].get_weights()[0],sum(abs_grads[tensors.index(layer)])) for layer in tensors]
    weights=[model.layers[layer].get_weights()[0] for layer in tensors]
    for idx,layer in enumerate(tensors):
      masks[idx].reshape(-1)[cs[idx].reshape(-1).argsort()[:int(sparsity*len(weights[idx].reshape(-1)))]]=0.
    return masks

  def prune_snip_global(self,model,tensors,sparsity,batch_X,batch_y):
    shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
    counts=[0]
    for layer in tensors:
      counts.append(counts[-1]+np.prod(model.layers[layer].get_weights()[0].shape))
    masks=np.concatenate([np.ones(model.layers[layer].get_weights()[0].shape).reshape(-1) for layer in tensors])
    with tf.GradientTape(persistent=True) as tape:
      output=model(batch_X)
      loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(output,batch_y))
    abs_grads=[np.abs(tape.gradient(loss,model.layers[layer].trainable_weights)[0].numpy()) for layer in tensors]
    cs=np.concatenate([np.divide(abs_grads[tensors.index(layer)]*model.layers[layer].get_weights()[0],sum(abs_grads[tensors.index(layer)])).reshape(-1) for layer in tensors])
    masks[cs.argsort()[:int(sparsity*counts[-1])]]=0.
    masks=[masks[counts[layer]:counts[layer+1]].reshape(shapes[layer]) for layer in range(len(tensors))]
    return masks

  def prune_random_layer(self,model,tensors,sparsities=[0.9238,0.725,0.57]):
    masks=[np.ones(model.layers[layer].get_weights()[0].shape) for layer in tensors]
    inds=[np.random.choice(range(len(mask.reshape(-1))),size=int(sparsities[ind]*len(mask.reshape(-1))),replace=False) for ind,mask in enumerate(masks)]
    for ind,mask in enumerate(masks):
      mask.reshape(-1)[inds[ind]]=0.
    return masks

  def prune_random_global(self,model,tensors,sparsity):
    shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
    counts=[0]
    for layer in tensors:
      counts.append(counts[-1]+np.prod(model.layers[layer].get_weights()[0].shape))
    masks=np.concatenate([np.ones(model.layers[layer].get_weights()[0].shape).reshape(-1) for layer in tensors])
    inds=np.random.choice(range(counts[-1]),size=int(sparsity*counts[-1]),replace=False)
    masks[inds]=0.
    masks=[masks[counts[layer]:counts[layer+1]].reshape(shapes[layer]) for layer in range(len(tensors))]
    return masks



  