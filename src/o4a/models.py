"""Neural network models shared across methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# ALIGNMENT NETWORK  (used by Hybrid, FullNonLinear, ReducedNonLinear)
# ==================================================================

class AlignmentNetwork(nn.Module):
    """Residual MLP: projects source space → target space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        # zero-init last linear for clean residual at start
        nn.init.zeros_(self.net[-2].weight)
        if self.net[-2].bias is not None:
            nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.input_proj(x)
        return proj + self.net(proj)


# ==================================================================
# MIXED LOSS  (MSE + cosine)
# ==================================================================

class MixedLoss(nn.Module):
    def __init__(self, alpha: float = 0.01, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(pred, target) + self.beta * (1.0 - F.cosine_similarity(pred, target, dim=1).mean())


# ==================================================================
# MLP PROBER  (used by FullNonLinear, ReducedNonLinear)
# ==================================================================

class MLPProber(nn.Module):
    """Binary classifier with two hidden layers."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) > 0.5).long()


# ==================================================================
# AUTOENCODER  (used by ReducedNonLinear)
# ==================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ==================================================================
# ENCODER + CLASSIFICATION HEAD  (used by OneForAll)
# ==================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) > 0.5).long()
