import torch
import torch.nn.functional as F
from torch import nn

class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, x):
        # q_phi(z|x)
        pass

    def decode(self, x):
        # p_theta(x|z)
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    x = torch.randn(1, 784)
    vae = VariationalAutoEncoder()
    print(vae(x).shape)