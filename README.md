# Variational Auto Encoder (VAE)

## Purpose:
This repository is an implementionation of the `Auto-Encoding Variational Bayes by Diederik P. Kingma and Max Welling` paper from the scratch using `PyTorch` to get a good understanding on the concepts used in `Variational Auto Encoder`. The MNIST dataset is used in the code for understanding the concepts.

## Scripts:
- `model.py`: `Encoder`, `Decoder` layers are defined and implemented for the `VAE`
- `train.py`: To train the VAE model from scratch

## Concepts:
Let's understand the key concepts involved in the `VAE` along with the code.

### Purpose of VAE:

A `VAE` is a type of generative model that consists of an `Encoder` network and `Decoder` network, similar to traditional Auto Encoders. The objective of the VAEs is information reconstruction and generation. For the given dataset sampled from unknown distribution we can conditionally generate new data with the same distribution.

### Encoder Architecture:

- The `Encoder` is responsibile to convert the Input Image to the `Latent Space Distribution`
- The `Encoder` network in `VAE` maps the input data to a probability distribution in the `latent space`, represented by a `mean` and `variance`.
- In our example code we have used `(4, 784)` -> This represents the `batch_size` as `4` and the `input image` dimension as `28*28 = 784`.
- A `Linear` layer is constructed with the shape of (`input_image_dimension`, `hidden_dimension`). Here the `hidden_dimension` is the dimension of the `hidden layers` present in the `Encoder` block.
- Two more `Linear` layers are used to calculate the `mean` and `variance`
- These parameters `mean` and `standard deviation` are returned from the `Encoder` to the `Decoder`

### Latent Space Distribution:

It is a probability distribution that represents the possible values or states that a hidden or latent variable can take on. This hidden variable captures the important information about the data in a compressed and structured form.

#### Example:

Suppose the `VAE` model is trained to generate image of cats. In this case, the `latent distribution` is like a set of instructions or rules for creating different aspects of a cat, such as it's shape, fur color, eye size etc. These instructions are probablistic, they won't give  a single fixed answer, rather a range of possibilities for each aspect.

The `Latent distribution` is a Gaussian distribution with two main parameters:

- `Mean (μ)`: This represents the center or average value of the distribution.
- `Variance (σ^2)`: This represents how spread out or uncertain the values are around the mean

When the sampling is done from the Gaussian Latent distribution, it is essentially generating a point in a multidimensional space, where each dimension corresponds to a different aspect of the data. Each sample is a set of values that define how the cat should look, along with this the degree of randomness is triggered by `variance`.

During training, the VAE learns to adjust the `mean` and `variance` of this latent distribution based on the input data.

### Reparametrization:

A key technique used in VAE to make the training process smoother and enable the model to learn the latent distribution effectively.

#### What is the problem with the traditional sampling in VAE?

In Traditional sampling, we directly sample from the Gaussian distribution by generating random numbers and scaling them by the learned `mean` and `standard deviation`

            `mean + random_number * standard_deviation`

This can be challenging to backpropogate gradients while training the model. To overcome this problem, `Reparametrization` technique is applied.

#### Reparametrization technique:

Instead of directly sampling from the distribution, reparametrization separates the randomness from the distribution. A random number is from a Standard Gaussian distribution, and then this sampled number is used to adjust the parameters mean and standard deviation of the latent distirbution.

                            epsilon = torch.randn_like(sigma)
                            z_reparametrized = mu + sigma*epsilon
                            x_reconstructed = self.decode(z_reparametrized)


From the above code, 



### Decoder Architecture:

- The `Decoder` is responsible to reconstruct the `input` from the 


Variational Auto Encoder implemented from scratch

- What is Variational Auto Encoder? and what is the purpose of building one?
- Why Binary Cross Entropy loss is used for the Auto Encoders?
- KL Divergence's purpose and its explanation
- Reparametrization
- Distribution
