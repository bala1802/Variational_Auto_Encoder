import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets #To access the MNIST dataset
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm

from model import VariationalAutoEncoder

'''
Configuration
'''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR_RATE = 3e-4

'''
Dataset Loading
'''
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

'''
Model Initialization
'''
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

'''
Reconstruction - Loss Function
'''
loss_fn = nn.BCELoss(reduction="sum")

'''
Model Training
'''

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))

    for i, (x, _) in loop:
        '''
        Forward Pass
        '''
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM) #x.shape[0] is The Number of examples, and the INPUT_DIM is 28*28 = 784 
        x_reconstructed, mu, sigma = model(x)

        '''
        Compute Loss
        '''
        reconstructed_loss = loss_fn(x_reconstructed, x) #To reconstruct the image
        kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) #Standard Gaussian

        '''
        Backpropogation
        '''
        loss = reconstructed_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


'''
Inferencing
'''
def inference(digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

for i in range(0, 10):
    inference(digit=i, num_examples=5)