import torch
from torch import nn


class ContactEnc(nn.Module):
    def __init__(self, dim_in, dim_h, dim_z):
        super().__init__()
        self.dim_h = dim_h
        self.dim_z = dim_z
        self.dim_in = dim_in

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 2 * self.dim_h),
            nn.SiLU(),
            nn.Linear(2 * self.dim_h, self.dim_h),
            nn.SiLU(),
            nn.Linear(self.dim_h, self.dim_z),
        )

    def forward(self, x):
        return self.encoder(x)


class ContactEncCLIP(nn.Module):
    def __init__(self, dim_in_clip, dim_h, dim_z):
        super().__init__()
        self.dim_h = dim_h
        self.dim_z = dim_z
        self.dim_in_clip = dim_in_clip

        self.encoder_clip = nn.Sequential(
            nn.Linear(dim_in_clip, 2 * self.dim_h),
            nn.SiLU(),
            nn.Linear(2 * self.dim_h, 2 * self.dim_h),
            nn.SiLU(),
            nn.Linear(2 * self.dim_h, self.dim_h),
            nn.SiLU(),
            nn.Linear(self.dim_h, self.dim_z),
        )

    def forward(self, x):
        return self.encoder_clip(x)


class ContactDec(nn.Module):
    def __init__(self, dim_in, dim_h, dim_z):
        super().__init__()
        self.dim_h = dim_h
        self.dim_z = dim_z
        self.dim_in = dim_in

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_h),
            nn.SiLU(),
            nn.Linear(self.dim_h, 2 * self.dim_h),
            nn.SiLU(),
            nn.Linear(2 * self.dim_h, dim_in)
        )

    def forward(self, z):
        return self.decoder(z)