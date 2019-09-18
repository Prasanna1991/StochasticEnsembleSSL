import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, zdim = 60):
        super(VAE, self).__init__()

        #encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2_mu = nn.Linear(256, zdim)
        self.fc2_logvar = nn.Linear(256, zdim)

        self.fc3 = nn.Linear(60, 128 * 4 * 4)

        #decoder
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))

        x = x.view(-1, 128 * 2 * 2)
        x = self.act(self.fc1(x))
        return self.fc2_mu(x), self.fc2_logvar(x)

    def decode(self, z):
        z = self.act(self.fc3(z))
        z = z.view(-1, 128, 4, 4)
        z = self.act(self.deconv1(z))
        z = self.act(self.deconv2(z))
        z = self.act(self.deconv3(z))
        z = self.act(self.deconv4(z))
        z = self.deconv5(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

class EnsembleClassifier(nn.Module):
    def __init__(self, classCount, zdim = 60):
        super(EnsembleClassifier, self).__init__()

        self.classifier_ = nn.Sequential(
            nn.Dropout(),
            nn.Linear(zdim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classCount),
        )

    def classifier(self, z):
        return torch.sigmoid(self.classifier_(z))