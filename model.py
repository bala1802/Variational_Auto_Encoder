import torch
import torch.nn.functional as F
from torch import nn


'''
Overall Architecture in the Variational Auto Encoders is:

Input Image => Convert that to Hidden dimension 
            => Mean and Standard deviations are calculated
            => Parametrization Trick
            => Decoder
            => Output Image is generated

'''
class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_dim, h_dim=200, z_dim=20):
        '''
            - input_dim: Dimension of the input image
            - h_dim: Dimension of the hidden layers in Encoder and the Decoder
            - z_dim: Dimension of the latent space

        '''
        
        super().__init__()

        '''
        Encoder
        '''
        self.img_to_hidden_dimension = nn.Linear(input_dim, h_dim)
        #Ensuring that the latent space is Standard Gaussian
        self.hidden_dim_to_mu = nn.Linear(h_dim, z_dim)
        self.hidden_dim_to_sigma = nn.Linear(h_dim, z_dim)

        '''
        Decoder
        '''
        self.z_dim_to_hidden_dim = nn.Linear(z_dim, h_dim)
        self.hidden_dim_to_img = nn.Linear(h_dim, input_dim)

        '''
        Activation function
        '''
        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_to_hidden_dimension(x))
        '''
            - Activation functions (ReLU) is not applied to calculate the mean and standard-deviation
            - Because the the mean or standard deviation might be in negative
        '''
        mu, sigma = self.hidden_dim_to_mu(h), self.hidden_dim_to_sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_dim_to_hidden_dim(z))
        return torch.sigmoid(self.hidden_dim_to_img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        
        '''
            - x_reconstructed -> Is for calculating the reconstructed loss
            - mu & sigma -> Is for the KL Divergence

        '''
        return x_reconstructed, mu, sigma


'''
if __name__ == "__main__":
    #The Input dimension for MNIST is 28X28 = 784
    x = torch.randn(4, 784) 
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)

    print("The reconstructed image, x_reconstructed Shape is : ", x_reconstructed.shape)
    print("The mean mu shape : ", mu.shape)
    print("The standard deviation sigma shape : ", sigma.shape)
'''