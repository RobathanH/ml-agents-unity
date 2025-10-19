from typing import Tuple

from mlagents.torch_utils import torch, nn


class VectorVAE(nn.Module):
    """
    Simple MLP VAE for vector observations. For inference/export, use encode_mean() to
    produce deterministic features. The forward() returns (recon, mu, logvar).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_size: int,
        hidden_layers: int = 2,
    ):
        super().__init__()
        enc_layers = []
        last_size = input_dim
        for _ in range(max(0, hidden_layers)):
            enc_layers.append(nn.Linear(last_size, hidden_size))
            enc_layers.append(nn.ReLU())
            last_size = hidden_size
        self.encoder = nn.Sequential(*enc_layers) if enc_layers else nn.Identity()
        self.mu = nn.Linear(last_size, latent_dim)
        self.logvar = nn.Linear(last_size, latent_dim)

        dec_layers = []
        last_dec = latent_dim
        for _ in range(max(0, hidden_layers)):
            dec_layers.append(nn.Linear(last_dec, hidden_size))
            dec_layers.append(nn.ReLU())
            last_dec = hidden_size
        dec_layers.append(nn.Linear(last_dec, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        clamped_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * clamped_logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Clamp logvar for stability in KL and sampling symmetry
        clamped_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if loss_type == "smooth_l1":
            recon_loss = torch.nn.functional.smooth_l1_loss(
                recon_x, x, beta=huber_delta, reduction="mean"
            )
        else:
            recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")
        # KL divergence between N(mu, sigma) and N(0, 1)
        kl = -0.5 * torch.mean(1 + clamped_logvar - mu.pow(2) - clamped_logvar.exp())
        total = recon_loss + beta * kl
        return total, recon_loss, kl


