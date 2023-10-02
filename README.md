# Variational_Auto_Encoder (VAE)

## Purpose:
This repository is an implementionation of the `Auto-Encoding Variational Bayes by Diederik P. Kingma and Max Welling` paper from the scratch using `PyTorch` to get a good understanding on the concepts used in `Variational Auto Encoder`. The MNIST dataset is used in the code for understanding the concepts.

## Scripts:
- `model.py`: Encoder, Decoder layers are defined and implemented for the VAE
- `train.py`: To train the VAE model from scratch

## Concepts:
Before diving into the configurations, and the implementation part, let's understand the key concepts involved in the `VAE`

### Purpose of VAE:

- A VAE is a type of generative model that consists of an `Encoder` network and `Decoder` network, similar to traditional Auto Encoders. The objective of the VAEs is information reconstruction and generation. For the given dataset sampled from unknown distribution we can conditionally generate new data with the same distribution. 

Variational Auto Encoder implemented from scratch

- What is Variational Auto Encoder? and what is the purpose of building one?
- Why Binary Cross Entropy loss is used for the Auto Encoders?
- KL Divergence's purpose and its explanation
- Reparametrization
- Distribution