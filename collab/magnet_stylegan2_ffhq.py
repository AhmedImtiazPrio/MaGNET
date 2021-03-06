# -*- coding: utf-8 -*-
"""MaGNET-StyleGAN2 FFHQ

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C6K77e7pGSGKQiiq7EDv_mJ5dUpBh-h5
"""

## Code for "MaGNET: Uniform Sampling from Deep Generative Network Manifolds without Retraining"
## https://arxiv.org/abs/2110.08009
## Authors: Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk
## MIT License
## Copyright (c) 2022 Ahmed Imtiaz Humayun

"""# MaGNET Sampling for StyleGAN2-FFHQ"""

# Commented out IPython magic to ensure Python compatibility.
#@title Download Model Weights

# %tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)
import numpy as np

!pip install --upgrade --no-cache-dir gdown

!git clone https://github.com/NVlabs/stylegan2
# !wget https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl -O ./stylegan2/stylegan2-ffhq-config-f.pkl
!wget https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-e.pkl -O ./stylegan2/stylegan2-ffhq-config-e.pkl

## Download precomputed volume scalars for FFHQ StyleGAN2e Pixelspace
!gdown https://drive.google.com/uc?id=1rhz2ymIqRHe4A_n3H4jVk1ZEB6EG7zP2 -O ./stylegan2/stylegan2_confige_ffhq_pixelspace.npz

## Download precomputed volume scalars for FFHQ StyleGAN2e Stylespace
# !gdown https://drive.google.com/uc?id=1fg6ealYoWhe0_MZJYbbG6U_hxJJh8Ahi -O ./stylegan2/stylegan2_confige_ffhq_stylespace.npz

## Download precomputed volume scalars for FFHQ StyleGAN2f Pixelspace
# !gdown https://drive.google.com/uc?id=1SSn-Vs59U0XoUcK8hCoK7bwFMFGtwI7q -O ./stylegan2/stylegan2_configf_ffhq_pixelspace.npz

#@title Import Utility Functions

def load_svds(svd_path,nsvds=250000):
  '''
  Import precomputed svds and corresponding latents
  '''
  with np.load(svd_path) as data:
    svds = data['svds'].reshape(-1,data['svds'].shape[-1])[:nsvds].astype(np.float64)
    latents = data['latents'].reshape(-1,data['latents'].shape[-1])[:nsvds]

  return latents,svds

def get_magnet_latents(num_samples,latents,svds,top_k=30,seed=None,replace=False,verbose=False):
  '''
  Sample latents uniformly via MaGNET sampling
  '''
  ## calculate volume scalars using top_k singular values
  detz = np.exp(np.log(svds[:,:top_k]).sum(1))
  detz = np.clip(detz,1e-200,1e200)

  if verbose:
    print('Volume Scalar range ',detz.min(),detz.max())
    
  proba = detz
  np.random.seed(seed)
  idx = np.random.choice(latents.shape[0], size=num_samples, p=proba / proba.sum(), replace=replace)
  return latents[idx]


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

def to_uint8(img):
  '''
  Convert image to uint8
  '''
  img = img*127.5 + 128
  img = np.clip(img,0,255)
  return img.astype(np.uint8)

import sys
sys.path.append('./stylegan2')

import pickle
import PIL.Image
import dnnlib.tflib as tflib
import matplotlib.pyplot as plt
import tqdm

tf.InteractiveSession()

config = 'e'             # to change the config/dataset/manifold please  
dataset = 'ffhq'         # change the download links in the 'download 
manifold = 'pixelspace'  # model weights' cell above.


with open(f'./stylegan2/stylegan2-ffhq-config-{config}.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

latents,svds = load_svds(f'./stylegan2/stylegan2_config{config}_{dataset}_{manifold}.npz')

sess = tf.get_default_session()

def generate(z,trunc=1,batch_size=10):
  '''
  batch generate images
  '''
  images = []

  for z_in in z.reshape(-1,batch_size,z.shape[-1]):
    img = Gs.get_output_for(z_in,np.zeros((z_in.shape[0],0)),truncation_psi=trunc)
    img = sess.run(img)
    images.append(to_uint8(img.transpose(0,2,3,1)))

  return np.vstack(images)

def create_magnet_montage(rows,cols,truncations,
                         top_k=30,seed=None):
  '''
  Create grid with truncation swept images with both naive and  magnet samples
  '''
  if type(truncations) == float:
    truncations = [truncations]

  cols_per_trunc = cols//len(truncations)
  
  trunc_grid = []
  for i,trunc in tqdm.tqdm(enumerate(truncations),total=len(truncations)):
      
      ## get naive samples
      np.random.seed(seed)
      z = np.random.randn(rows//2*cols_per_trunc,latents.shape[-1])
      imgs = generate(z,trunc,batch_size=rows)
      imgs1 = imgrid(imgs,cols=cols_per_trunc,pad=10,pad_value=255)

      ## get magnet samples
      z = get_magnet_latents(
          num_samples=rows//2*cols_per_trunc,
          latents=latents,svds=svds,
          top_k=top_k,seed=seed
                          )

      imgs = generate(z,trunc,batch_size=rows)
      imgs2 = imgrid(imgs,cols=cols_per_trunc,pad=10,pad_value=0)

      trunc_grid.append(imgrid(np.asarray([imgs1,imgs2]),
                               cols=1,
                               pad=30,pad_value=0
                               )
      )
      
  return np.stack(trunc_grid)

rows = 10
cols = 20
truncations = [.3,.5,.7,.9,1]

trunc_grid = create_magnet_montage(
              rows=rows,
              cols=cols,
              truncations=truncations
          )

plt.figure(figsize=(27,30))

plt.title(f"From left to right, every {cols//len(truncations)} column contains images with naive sampling (top {rows//2} rows) and magnet sampling (bottom {rows//2} rows) with truncation $\in$ {truncations}",fontsize="15")

plt.imshow(imgrid(trunc_grid,cols=trunc_grid.shape[0],pad=30,pad_value=255))

plt.xticks([]);
plt.yticks([]);

