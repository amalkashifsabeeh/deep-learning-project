"""A small convolutional autoencoder for inpainting, with a bottleneck I can change the size of."""
import torch
import torch.nn as nn


class InpaintAE(nn.Module):
    """The input is the corrupted image plus the mask as a second channel.

    The image goes from 28x28 down to 14, 7 and 4, then is flattened into a latent vector,
    and the decoder mirrors the encoder back up. The output is between 0 and 1.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim  # the bottleneck size; I want to vary it later
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),    # 14x14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 7x7
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 4x4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),  # the bottleneck: everything the decoder knows comes through here
        )
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 64 * 4 * 4), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0), nn.ReLU(),   # 7x7
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),   # 14x14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),                # 28x28
            nn.Sigmoid(),  # pixels are between 0 and 1, so the output is too
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x has shape (B, 1, 28, 28). mask has the same shape, as floats, with 1 meaning missing."""
        # I zero out the hole pixels so the model never sees what was there, whatever x holds in the hole
        corrupted = x * (1.0 - mask)
        # the mask goes in as a second channel, so a hole is not confused with a real black pixel
        z = self.encoder(torch.cat([corrupted, mask], dim=1))
        h = self.decoder_fc(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    @torch.no_grad()
    def inpaint(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The prediction with the visible pixels pasted back in. This is the version I evaluate."""
        out = self.forward(x, mask)
        return mask * out + (1.0 - mask) * x


def inpaint_loss(out: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, w_visible: float = 0.1):
    """MSE on the hole pixels, which is what I care about, plus a small weight on the visible pixels."""
    sq = (out - target) ** 2
    # clamp(min=1) avoids dividing by zero when a mask has no hole pixels (or no visible ones)
    hole = (sq * mask).sum() / mask.sum().clamp(min=1.0)
    vis = (sq * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1.0)
    return hole + w_visible * vis, hole.detach()
