import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, attr_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim, attr_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class MLPClassifier(nn.Module):
    def __init__(self, x_dim, attr_dim, out_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + attr_dim, 2000),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2000, 1200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, 1200),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, out_dim),
        )

    def forward(self, x):
        return self.model(x)
