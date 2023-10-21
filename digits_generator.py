import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST 
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
    
def get_generator_block(input_dim, output_dim):
    """
    Function for returning a block of the generator's neural network
    given input and output dimesions
    Args:
        input_dim (int): the dimesion of the input vector
        output_dim (int): the dimension of the output vector
    Returns:
        a generator neural network layer, with a linear transformation 
        followed by a batch normalizaation and then a relu activation
    """
    
    return nn.Sequential(
    nn.Linear(input_dim, output_dim),
    nn.BatchNorm1d(output_dim),
    nn.ReLU(inplace=True)
    )
    
class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimesion, a scalar
    """
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim*2),
            get_generator_block(hidden_dim*2, hidden_dim*4),
            get_generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid(),
        )
        
    def forward(self, noise):
        return self.gen(noise)
    
    def get_gen(self):
        return self.gen 
    
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def get_discriminator_block(input_dim, output_dim):
    """
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions
    Parameters:
        input_dim: the dimension of the input vector
        output_dim: the dimension of the output vector
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2),
    )
    
class Discriminator(nn.Module):
    """
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim*4),
            get_discriminator_block(hidden_dim*4, hidden_dim*2),
            get_discriminator_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, image):
        return self.disc(image)
    
    def get_disc(self):
        return self.disc
    

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    Return the loss of the discriminator given inputs
    """
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise).detach()
    pred = disc(fake)
    loss1 = criterion(pred, torch.zeros_like(pred, device=device))
    pred2 = disc(real)
    loss2 = criterion(pred, torch.ones_like(pred2, device=device))
    disc_loss = (loss1 + loss2)/2
    return disc_loss 

def get_gen_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    Return the loss of the Generator given inputs
    """
    noise = get_noise(num_images, z_dim, device)
    fake = gen(noise)
    pred = disc(fake)
    gen_loss = criterion(pred, torch.ones_like(pred))
    return gen_loss



def main():
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
    
    dataloader = DataLoader(
        MNIST('.', download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True
    )
    
    device= 'cpu'
    
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    gen_loss = False
    for epoch in range(n_epochs):
        
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            
            disc_loss.backward()
            disc_opt.step()
            
                
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()
            
            mean_discriminator_loss += disc_loss.item() / display_step 
            
            mean_generator_loss += gen_loss.item() / display_step 
            
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
                
            
    
if __name__ == "__main__":
    main()
