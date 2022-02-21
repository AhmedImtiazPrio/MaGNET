## MaGNET: Uniform Sampling from Deep Generative Network Manifolds Without Retraining
### Official repository for ICLR 2022 paper

Paper Link: https://arxiv.org/abs/2110.08009

Abstract: _Deep Generative Networks (DGNs) are extensively employed in Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and their variants to approximate the data manifold and distribution. However, training samples are often distributed in a non-uniform fashion on the manifold, due to costs or convenience of collection. For example, the CelebA dataset contains a large fraction of smiling faces. These inconsistencies will be reproduced when sampling from the trained DGN, which is not always preferred, e.g., for fairness or data augmentation. In response, we develop MaGNET, a novel and theoretically motivated latent space sampler for any pre-trained DGN, that produces samples uniformly distributed on the learned manifold. We perform a range of experiments on various datasets and DGNs, e.g., for the state-of-the-art StyleGAN2 trained on FFHQ dataset, uniform sampling via MaGNET increases distribution precision and recall by 4.1% & 3.0% and decreases gender bias by 41.2%, without requiring labels or retraining. As uniform distribution does not imply uniform semantic distribution, we also explore separately how semantic attributes of generated samples vary under MaGNET sampling._

| Google Collabs | &nbsp;
| :---- | :----
| MaGNET-Stylegan2 | Link
| MaGNET-BigGAN | Link
| MaGNET-ProGAN | Link
| MaGNET-NVAE | Link

### Requirements

Since MaGNET is a plug and play method, initially we are making separate google collabs for Tensorflow and Pytorch implementations of StyleGAN2 (TF), BigGAN (TF), ProGAN (TF) and NVAE (Pytorch). The google collab code uses precomputed volume scalars to perform MaGNET sampling. We will also be adding submodules into this repo as plug and play examples for Tensorflow(=>1.15) and Pytorch(>=1.5), with methods to compute the volume scalars and perform metric calculation.

```
tensorflow-gpu=>1.15
# or
pytorch>=1.5
```


### Additional Materials

* [Official Google Drive]()
    * [Samples]()
        * [StyleGAN2-FFHQ Uncurated 5K with Face Attribute JSONs]()
    * [SVDs]()
        * [StyleGAN2 config-e FFHQ StyleSpace]()
        * [StyleGAN2 config-e FFHQ PixelSpace]()
        * [StyleGAN2 config-f FFHQ PixelSpace]()
        * [ProGAN CelebAHQ PixelSpace-256]()
        * [BigGAN-deep Imagenet 256]()



