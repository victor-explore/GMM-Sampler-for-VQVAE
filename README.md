# GMM-Sampler-for-VQVAE

This repository contains an implementation of a Vector Quantized Variational Autoencoder (VQ-VAE) with a Gaussian Mixture Model (GMM) sampler. The model combines the discrete latent space of VQ-VAE with GMM sampling to better capture complex data distributions.

## Overview

The VQ-VAE architecture consists of:
- An encoder that maps input data to discrete latent codes
- A codebook that stores learnable embedding vectors
- A decoder that reconstructs the input from the quantized embeddings

Key features:
- Codebook dimension: 128
- Number of codebook vectors: 1024 
- Commitment cost: 1
- Training for 50 epochs
- Batch size of 64
- Learning rate of 0.001

## Training Process

The model is trained on image data with:
- Per-batch loss monitoring
- Regular saving of reconstruction images
- Model checkpointing every epoch
- Visualization of training progress

The training logs show the loss decreasing from ~2.0 in early epochs to ~0.4 in later epochs, indicating successful learning of the data distribution.

## Usage

To train the model:
1. Install the required dependencies (PyTorch, torchvision)
2. Run the Jupyter notebook GMM-Sampler-for-VQVAE.ipynb
3. Training progress and reconstructions will be saved to:
   - reconstructed_images/ directory
   - vqvae_models/ directory

## Advantages

This implementation addresses limitations of standard VAEs by:
- Using discrete latent codes instead of continuous distributions
- Employing GMM sampling to better model multi-modal data
- Avoiding the restrictive standard normal prior assumption

## Results

The model demonstrates:
- Stable training with consistent loss reduction
- High quality image reconstructions
- Effective discrete latent space learning
- Improved modeling of complex data distributions

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Jupyter Notebook
