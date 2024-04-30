import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Encoder
class Encoder(nn.Module):
    def __init__(self, in_dim, class_embedding_dim, latent_dim):
        super(Encoder,self).__init__()
        self.fc1=nn.Linear(in_dim + class_embedding_dim, 4096)
        self.fc3=nn.Linear(4096, latent_dim*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_dim*2, latent_dim)
        self.linear_log_var = nn.Linear(latent_dim*2, latent_dim)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


# Decoder/Generator
class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, out_dim):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(z_dim + c_dim, 4096)
        self.fc3 = nn.Linear(4096, out_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x
       

# conditional discriminator
class Discriminator(nn.Module):
    def __init__(self, c_dim): 
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2048 + c_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h


# classifier
class Myclassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(Myclassifier, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o