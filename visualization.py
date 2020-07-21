import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_mask(samples,folder_path):
  masks=np.stack([np.load(folder_path+f'/{s}_mask.npy') for s in range(samples)])
  mask=np.mean(masks,axis=0)
  image=np.mean(mask,axis=1)*255
  cv2.imwrite(folder_path+'_mask_image.png',image.reshape((28,28)))

def prepare_curves(fibers):
  length=min([len(fibers[i]) for i in range(len(fibers))])
  avg_fiber=np.zeros(length)
  min_fiber=np.ones(length)
  max_fiber=np.zeros(length)
  for fiber in fibers:
    fiber=fiber[:length]
    avg_fiber+=np.divide(fiber,len(fibers))
    min_fiber=np.minimum(min_fiber,fiber)
    max_fiber=np.maximum(max_fiber,fiber)
  return min_fiber,avg_fiber,max_fiber

def show_curves(samples,architecture,dataset):
  iterations=np.load('results/'+architecture+'/snip_global/0_iterations.npy')
  accuracies_snip_global=[np.load('results/'+architecture+'/snip_global/{}_accuracies.npy'.format(sample)) for sample in range(samples)]
  accuracies_snip_layer=[np.load('results/'+architecture+'/snip_layer/{}_accuracies.npy'.format(sample)) for sample in range(samples)]
  accuracies_random_layer=[np.load('results/'+architecture+'/random_layer/{}_accuracies.npy'.format(sample)) for sample in range(samples)]
  accuracies_random_global=[np.load('results/'+architecture+'/random_global/{}_accuracies.npy'.format(sample)) for sample in range(samples)]
  accuracies_base=[np.load('results/'+architecture+'/base/{}_accuracies.npy'.format(sample)) for sample in range(samples)]
  snip_global_band=prepare_curves(accuracies_snip_global)
  snip_layer_band=prepare_curves(accuracies_snip_layer)
  random_layer_band=prepare_curves(accuracies_random_layer)
  random_global_band=prepare_curves(accuracies_random_global)
  base_band=prepare_curves(accuracies_base)
  snip_global={'min':snip_global_band[0],'avg':snip_global_band[1],'max':snip_global_band[2]}
  snip_layer={'min':snip_layer_band[0],'avg':snip_layer_band[1],'max':snip_layer_band[2]}
  random_global={'min':random_global_band[0],'avg':random_global_band[1],'max':random_global_band[2]}
  random_layer={'min':random_layer_band[0],'avg':random_layer_band[1],'max':random_layer_band[2]}
  base={'min':base_band[0],'avg':base_band[1],'max':base_band[2]}
  plt.plot(iterations,snip_global['avg'],color='blue',label='snip global')
  plt.fill_between(iterations,snip_global['min'],snip_global['max'],color='blue',alpha=0.2)
  plt.plot(iterations,snip_layer['avg'],color='purple',label='snip layer')
  plt.fill_between(iterations,snip_layer['min'],snip_layer['max'],color='purple',alpha=0.2)
  plt.plot(iterations,random_global['avg'],color='turquoise',label='random global')
  plt.fill_between(iterations,random_global['min'],random_global['max'],color='turquoise',alpha=0.2)
  plt.plot(iterations,random_layer['avg'],color='green',label='random layer')
  plt.fill_between(iterations,random_layer['min'],random_layer['max'],color='green',alpha=0.2)
  plt.plot(iterations,base['avg'],color='black',label='base')
  plt.fill_between(iterations,base['min'],base['max'],color='black',alpha=0.2)
  plt.title(architecture+' on '+dataset)
  plt.xlabel('iterations')
  plt.ylabel('test accuracy')
  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.show()

#show_curves(5,'lenet300100','mnist')
show_mask(5,'results/lenet300100/snip_layer')





