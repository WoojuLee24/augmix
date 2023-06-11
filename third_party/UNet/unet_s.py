import torch
import torch.nn as nn

class UNetS(nn.Module):
    def __init__(self, args, out_channels=3):
        super(UNetS, self).__init__()

        self.args = args
        self.out_channels = out_channels

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Encoder
        self.enc0 = CBR2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc1 = CBR2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc2 = CBR2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc3 = CBR2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.dec2 = CBR2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec1 = CBR2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc1 = self.pool(enc1)
        enc2 = self.enc2(enc1)
        enc2 = self.pool(enc2)
        enc3 = self.enc3(enc2)
        # enc3 = self.pool(enc3)

        cat3 = torch.cat((enc3, enc2), dim=1)
        dec2 = self.dec2(cat3)
        cat2 = torch.cat((self.upsample(dec2), enc1), dim=1)
        dec1 = self.dec1(cat2)
        cat1 = torch.cat((self.upsample(dec1), enc0), dim=1)
        logits = self.fc(cat1)

        return logits