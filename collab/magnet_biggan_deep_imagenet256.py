# -*- coding: utf-8 -*-
"""MaGNET-BigGAN-deep ImageNet256

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r6QX8jYe1tsQmZFO4myw3O29nqcKfdx5
"""

## Code for "MaGNET: Uniform Sampling from Deep Generative Network Manifolds without Retraining"
## https://arxiv.org/abs/2110.08009
## Authors: Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk
## Adapted from TFHUB BigGAN-deep-256 @ https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb

#@title Download Assets
!pip install --upgrade --no-cache-dir gdown

# download biggan-deep volume scalars
!gdown 'https://drive.google.com/uc?id=1HMkMYZDKN4fN6Ay1XS8A401kQH4VsqLE'
!gdown 'https://drive.google.com/uc?id=108lt0T3EwDfjtqUgRq26IpIbXxhsSfbg'
!gdown 'https://drive.google.com/uc?id=1s-yo-dmE83GZxSCUoIJLmPDY-6rz7nxk'
!gdown 'https://drive.google.com/uc?id=1FS0L-HeEK5JBG_p9IpMbV03Zrl7kLxrG'
!gdown 'https://drive.google.com/uc?id=1JG-XtpNZ9OKQamGm5voiywA2kA7Q5YZu'
!gdown 'https://drive.google.com/uc?id=1a9TBuQHqfzwIXVHNN-8fUDRS7zBzniBE'
!gdown 'https://drive.google.com/uc?id=1wtmCOSgI3cM1AslAH1IDpa8Se4pwJcBl'
!gdown 'https://drive.google.com/uc?id=17YWfv7pVOsqtoukNygd2bzil9iy_KdpR'
!gdown 'https://drive.google.com/uc?id=1A0nK5SrU9De8IfK57mCHbyPvS6sg_KKk'
!gdown 'https://drive.google.com/uc?id=1rWVNv7SOEFz74IGIX6nJKeXdiRxwtJyV'
!gdown 'https://drive.google.com/uc?id=1MP-5kWK68O71MA4NEwDxq6kfdzbaV5i2'
!gdown 'https://drive.google.com/uc?id=1G8qpyRY2LHNnrtKxuo9xiX2tKa4EBCMH'
!gdown 'https://drive.google.com/uc?id=1URyDhEAuXErGmCPWpvHFD9ELjXvZz7tD'
!gdown 'https://drive.google.com/uc?id=1Zj7wXLIdZX7nhBvaKRk-AjbwybD4CEuE'
!gdown 'https://drive.google.com/uc?id=1_xaQYB4TPgN6KUhBgSLDkRYq-cgje3nU'
!gdown 'https://drive.google.com/uc?id=19bNOFeBbEzVED6nm44dUfx4wvN9lxpUg'
!gdown 'https://drive.google.com/uc?id=1CnET0AJzcpN4EcO8PAxB8xhxMd1ejEkh'
!gdown 'https://drive.google.com/uc?id=1z1aS9rmmwfN0NHWMUg-C8k7lozrggcFy'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import io
import IPython.display
import numpy as np
import PIL.Image
from scipy.stats import truncnorm
import tensorflow_hub as hub
import tqdm
import matplotlib.pyplot as plt

## get tfhub biggan-deep-256

module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'  # 256x256 BigGAN-deep

## load biggan

tf.reset_default_graph()
print('Loading BigGAN module from:', module_path)
module = hub.Module(module_path)
inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}
output = module(inputs)

print()
print('Inputs:\n', '\n'.join(
    '  {}: {}'.format(*kv) for kv in inputs.items()))
print()
print('Output:', output)

input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']

dim_z = input_z.shape.as_list()[1]
vocab_size = input_y.shape.as_list()[1]

#@title Utility Functions

def truncated_z_sample(batch_size, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
  return truncation * values

def one_hot(index, vocab_size=vocab_size):
  index = np.asarray(index)
  if len(index.shape) == 0:
    index = np.asarray([index])
  assert len(index.shape) == 1
  num = index.shape[0]
  output = np.zeros((num, vocab_size), dtype=np.float32)
  output[np.arange(num), index] = 1
  return output

def one_hot_if_needed(label, vocab_size=vocab_size):
  label = np.asarray(label)
  if len(label.shape) <= 1:
    label = one_hot(label, vocab_size)
  assert len(label.shape) == 2
  return label

def sample(sess, noise, label, truncation=1., batch_size=8,
           vocab_size=vocab_size):
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot_if_needed(label, vocab_size)
  ims = []
  for batch_start in range(0, num, batch_size):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims.append(sess.run(output, feed_dict=feed_dict))
  ims = np.concatenate(ims, axis=0)
  assert ims.shape[0] == num
  ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
  ims = np.uint8(ims)
  return ims

def interpolate(A, B, num_interps):
  if A.shape != B.shape:
    raise ValueError('A and B must have the same shape to interpolate.')
  alphas = np.linspace(0, 1, num_interps)
  return np.array([(1-a)*A + a*B for a in alphas])

def imgrid(imarray, cols=5, pad=1):
  if imarray.dtype != np.uint8:
    raise ValueError('imgrid input imarray must be uint8')
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(a).save(data, format)
  im_data = data.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print(('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format))
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def load_svds(svd_path):
  '''
  Import precomputed svds and corresponding latents
  '''

  with np.load(svd_path) as data:
    svds = data['svds'].astype(np.float64).reshape(-1,data['svds'].shape[-1])
    latents = data['latents'].reshape(-1,data['latents'].shape[-1])

  assert svds.shape[0] == latents.shape[0]

  mask = svds.sum(1) == 0
  svds = svds[np.logical_not(mask)]
  latents = latents[np.logical_not(mask)]

  return latents,svds

def imgrid(imarray, cols=10, pad=1,pad_value=0):
  '''
  Display image array in a grid
  '''
  if imarray.dtype != np.uint8:
      raise ValueError('imgrid input imarray must be uint8')
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=pad_value)
  H += pad
  W += pad
  grid = (imarray
        .reshape(rows, cols, H, W, C)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows*H, cols*W, C))
  if pad:
      grid = grid[:-pad, :-pad]
  return grid

def get_magnet_latents(num_samples,latents,svds,top_k=30,seed=0):
    
    detz = np.exp(np.log(svds[:,:top_k]).sum(1))
        
    proba = detz
    proba = np.clip(proba,1e-200,1e200) # for numerical stability
    print('detz range ',proba.min(),proba.max())
    
    np.random.seed(seed)
    idx = np.random.choice(latents.shape[0], size=num_samples, p=proba / proba.sum(), replace=False)
    return latents[idx,:]


def to_uint8(img):
  '''
  Convert image to uint8
  '''
  img = img*127.5 + 128
  img = np.clip(img,0,255)
  return img.astype(np.uint8)

initializer = tf.global_variables_initializer()
sess = tf.Session()
sess.run(initializer)

#@title Category-conditional sampling { display-mode: "form", run: "auto" }

sampling_method = 'MaGNET' #@param ['MaGNET','naive']
num_samples = 20 #@param {type:"slider", min:1, max:20, step:1}
truncation = 0.9 #@param {type:"slider", min:0.02, max:1, step:0.02}
# truncation can be applied in multiple ways, for qualitative we are using variance scaling.
noise_seed = 0 #@param {type:"slider", min:0, max:100, step:1}
category = "933_cheeseburger" #@param ['0_tench','1_goldfish','10_brambling','12_house finch','15_robin','130_flamingo','207_golden retriever','254_pug','258_Samoyed','259_Pomeranian','281_tabby','285_Egyptian cat','291_lion','292_tiger','293_cheetah','387_lesser panda','933_cheeseburger']
y = int(category.split('_')[0])

if sampling_method=='naive':
  z = truncated_z_sample(num_samples,
                         truncation=1,
                         seed=noise_seed)  
else:
  latents,svds = load_svds(f'/content/{category}.npz')
  z = get_magnet_latents(num_samples, latents, svds, top_k=10, seed=noise_seed)
  
ims = sample(sess, z*truncation, y, truncation=1)
imshow(imgrid(ims, cols=min(num_samples, 10)))

def create_magnet_montage(categories,rows,cols,truncation,top_k=10,seed=None):

  if type(categories) == str:
    categories = [categories]

  cols_per_cat = cols//len(categories)

  cat_grid = []
  
  for i, category in tqdm.tqdm(enumerate(categories),total=len(categories)):

    y = int(category.split('_')[0])

    ## get naive samples
    z = truncated_z_sample(num_samples,
                         truncation=1,
                         seed=noise_seed)  
    
    imgs = sample(sess, z*truncation, y, truncation=1)
    imgs1 = imgrid(imgs,cols=cols_per_cat,pad=10,pad_value=255)

    ## get magnet samples
    latents,svds = load_svds(f'/content/{category}.npz')
    z = get_magnet_latents(num_samples, latents, svds, top_k=10, seed=noise_seed)

    imgs = sample(sess, z*truncation, y, truncation=1)
    imgs2 = imgrid(imgs,cols=cols_per_cat,pad=10,pad_value=0)

    cat_grid.append(imgrid(np.asarray([imgs1,imgs2]),
                               cols=1,
                               pad=30,pad_value=0
                               )
      )

  return np.asarray(cat_grid)

truncation = .9
categories = ['0_tench','130_flamingo','207_golden retriever','258_Samoyed','933_cheeseburger']

rows = 10
cols = 20

trunc_grid = create_magnet_montage(
              categories,
              rows=rows,
              cols=cols,
              top_k=10,
              truncation=truncation
          )

plt.figure(figsize=(27,30))

plt.title(f"From left to right, every {cols//len(categories)} column contains images with naive sampling (top {rows//2} rows) and magnet sampling (bottom {rows//2} rows) with classes $\in$ {[each.split('_')[-1] for each in categories]}",fontsize="15")

plt.imshow(imgrid(trunc_grid,cols=trunc_grid.shape[0],pad=30,pad_value=255))

plt.xticks([]);
plt.yticks([]);