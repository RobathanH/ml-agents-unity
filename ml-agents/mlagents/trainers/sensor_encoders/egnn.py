from typing import Optional, Tuple

from mlagents.torch_utils import torch, nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2):
        super().__init__()
        dims = [in_dim]
        for _ in range(max(0, layers - 1)):
            dims.append(hidden_dim)
        dims.append(out_dim)
        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.SiLU())
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGNNLayer(nn.Module):
    """
    A simplified EGNN layer. Given node positions x \in R^{B x N x 3} and node features h \in R^{B x N x H},
    computes messages over a learned neighborhood and updates both positions and features equivariantly.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        message_dim: int,
        k_neighbors: int = 8,
    ):
        super().__init__()
        self.k = int(k_neighbors)
        # Edge function phi_e([h_i, h_j, ||x_i - x_j||^2]) -> m_ij
        self.phi_e = MLP(feature_dim * 2 + 1, hidden_dim, message_dim, layers=2)
        # Node feature update phi_h([h_i, sum_j m_ij])
        self.phi_h = MLP(feature_dim + message_dim, hidden_dim, feature_dim, layers=2)
        # Position update scaling
        self.phi_x = MLP(message_dim, hidden_dim, 1, layers=2)

    @staticmethod
    def _pairwise_squared_dist(x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 3]
        # returns [B, N, N]
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        return (diff * diff).sum(-1)

    @staticmethod
    def _knn_indices(d2: torch.Tensor, k: int, eye: torch.Tensor) -> torch.Tensor:
        # d2: [B, N, N]; select k nearest neighbors using ONNX-friendly TopK
        # Use largest=True on negated distances to emulate ascending TopK.
        b, n, _ = d2.shape
        d2_neg = -d2
        # Strongly penalize self edges so they are never selected when taking largest
        d2_neg = d2_neg - eye.unsqueeze(0) * 1e9
        k_eff = min(k, max(1, n - 1))
        _, idx = torch.topk(d2_neg, k=k_eff, dim=-1, largest=True)
        return idx  # [B, N, k]

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
        eye: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, 3], h: [B, N, H], mask: [B, N] where 1.0 indicates padding to ignore
        # eye: optional [N, N] identity constant (see EGNNEncoder); passing it as a
        # precomputed buffer keeps EyeLike out of the ONNX graph (Sentis does not
        # support EyeLike).
        b, n, _ = x.shape
        if eye is None or eye.shape[0] != n:
            eye = torch.eye(n, device=x.device, dtype=x.dtype)
        d2 = self._pairwise_squared_dist(x)
        idx = self._knn_indices(d2, self.k, eye)
        # Gather neighbors via a one-hot matmul (rows of an identity looked up by
        # neighbor index -> single-tensor Gather). Multi-tensor advanced indexing
        # and shape-derived arange+matmul both crash the TorchScript ONNX exporter,
        # GatherElements needs opset >= 11 (models export at opset 9), and a
        # batch-offset arange would be baked as a constant at trace time, breaking
        # batched inference. N is static (BufferSensor pads to max entities), so
        # only the batch axis is dynamic.
        onehot = eye[idx]  # [B, N, K, N]
        x_j = torch.matmul(onehot, x.unsqueeze(1))  # [B, N, K, 3]
        h_j = torch.matmul(onehot, h.unsqueeze(1))  # [B, N, K, H]
        # For the "i" nodes, just expand along the neighbor dimension
        x_i = x.unsqueeze(2).expand(-1, -1, idx.shape[2], -1)  # [B, N, K, 3]
        h_i = h.unsqueeze(2).expand(-1, -1, idx.shape[2], -1)  # [B, N, K, H]
        # Compute edge messages
        r2 = ((x_i - x_j) * (x_i - x_j)).sum(-1, keepdim=True)
        e_ij = torch.cat([h_i, h_j, r2], dim=-1)
        m_ij = self.phi_e(e_ij)  # [B, N, K, M]

        # Position update
        scale = self.phi_x(m_ij)  # [B, N, K, 1]
        dx = (x_i - x_j) * scale
        dx = dx.sum(dim=2) / max(1, self.k)

        # Feature update
        m_sum = m_ij.sum(dim=2)
        h_in = torch.cat([h, m_sum], dim=-1)
        dh = self.phi_h(h_in)

        if mask is not None:
            inv_mask = (1.0 - mask).unsqueeze(-1)  # [B, N, 1]
            dx = dx * inv_mask
            dh = dh * inv_mask

        x_out = x + dx
        h_out = h + dh
        return x_out, h_out


class EGNNEncoder(nn.Module):
    """
    EGNN encoder over variable number of nodes. Accepts padded entity tensors and a mask; returns per-node embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_size: int,
        hidden_dim: int = 128,
        message_dim: int = 64,
        num_layers: int = 3,
        k_neighbors: int = 8,
        pos_dim: int = 3,
        max_entities: Optional[int] = None,
    ):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.attr_dim = max(0, int(input_dim) - self.pos_dim)
        self.embedding_size = int(embedding_size)
        # Identity constant for kNN self-edge masking and one-hot neighbor gather.
        # Registered as a (non-persistent) buffer so ONNX export emits a plain
        # initializer instead of EyeLike, which Sentis does not support.
        if max_entities is not None:
            self.register_buffer(
                "node_eye", torch.eye(int(max_entities)), persistent=False
            )
        else:
            self.node_eye = None
        self.input_proj = nn.Linear(self.attr_dim if self.attr_dim > 0 else 1, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    feature_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    message_dim=message_dim,
                    k_neighbors=k_neighbors,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, self.embedding_size)

    @staticmethod
    def mask_from_entities(entities: torch.Tensor) -> torch.Tensor:
        # Padding rows are all-zeros
        return (torch.sum(entities * entities, dim=-1) < 1e-4).float()

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        # entities: [B, N, D]
        b, n, d = entities.shape
        mask = self.mask_from_entities(entities)  # [B, N]
        pos = entities[..., : self.pos_dim]
        if self.attr_dim > 0:
            attrs = entities[..., self.pos_dim :]
        else:
            attrs = torch.ones(b, n, 1, device=entities.device, dtype=entities.dtype)
        # Project attrs to hidden features
        h = self.input_proj(attrs)
        for layer in self.layers:
            pos, h = layer(pos, h, mask, self.node_eye)
        out = self.out_proj(h)
        # Zero out padded nodes for cleanliness
        out = out * (1.0 - mask).unsqueeze(-1)
        return out  # [B, N, E]


