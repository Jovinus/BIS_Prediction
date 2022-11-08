import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, MFCC
from torchvision.transforms import Resize


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # |x| = (batch_size, in_channels, h, w)

        y = self.layers(x)
        # |y| = (batch_size, out_channels, h, w)

        return y


class ConvolutionalClassifier(nn.Module):

    def __init__(self, output_size):
        self.output_size = output_size

        super().__init__()
        
        self.preprocess_layer = nn.Sequential(
            MFCC(sample_rate=100),
            # Spectrogram(n_fft=100, hop_length=100),
            Resize(size=(60, 60))
        )
        

        self.blocks = nn.Sequential( # |x| = (n, 1, 28, 28)
            ConvolutionBlock(1, 32), # (n, 32, 14, 14)
            ConvolutionBlock(32, 64), # (n, 64, 7, 7)
            ConvolutionBlock(64, 128), # (n, 128, 4, 4)
            ConvolutionBlock(128, 256), # (n, 256, 2, 2)
            ConvolutionBlock(256, 512), # (n, 512, 2, 2)
            ConvolutionBlock(512, 512*2), # (n, 1024, 1, 1)
        )
        self.layers = nn.Sequential(
            nn.Linear(512*2, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        
        z = self.preprocess_layer(x)
        
        assert z.dim() > 2

        if z.dim() == 3:
            # |z| = (batch_size, h, w)
            z = z.view(-1, 1, z.size(-2), z.size(-1))
        # |z| = (batch_size, 1, h, w)

        z = self.blocks(z)
        # |z| = (batch_size, 512, 1, 1)
        
        y = self.layers(z.squeeze())
        # |y| = (batch_size, output_size)

        return y
