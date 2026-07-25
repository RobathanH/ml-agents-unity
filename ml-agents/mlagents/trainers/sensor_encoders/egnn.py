from typing import List, Optional, Tuple

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


def quat_to_axes(
    q: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unity quaternion (x, y, z, w) -> the three columns of its rotation matrix,
    i.e. the entity's local right / up / forward axes expressed in the sensor's
    virtual-root frame.

    Each column is an *equivariant* vector: under a global rotation R of the
    scene the quaternion becomes q_R * q and the columns become R * column.
    Feeding raw quaternion components into an invariant feature channel throws
    that structure away and forces the network to learn frame handling from
    scratch, which is what the pre-fix encoder did.

    A zero quaternion (an all-zero padding row) normalises to zero and yields
    the identity matrix rather than NaN.
    """
    n2 = (q * q).sum(-1, keepdim=True)
    q = q / torch.sqrt(n2.clamp_min(eps))
    x, y, z, w = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    right = torch.cat(
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)], dim=-1
    )
    up = torch.cat(
        [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)], dim=-1
    )
    forward = torch.cat(
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)], dim=-1
    )
    return right, up, forward


class EGNNLayer(nn.Module):
    """
    A simplified EGNN layer. Given node positions x \\in R^{B x N x 3} and node features h \\in R^{B x N x H},
    computes messages over a learned neighborhood and updates both positions and features equivariantly.

    When vector channels are supplied (see EGNNEncoder, attr_mode="equivariant")
    the edge function additionally receives the rotation-invariant contractions
    between the edge direction and each node's vector channels, which is what
    lets the message reason about relative geometry -- "is j above my foot",
    "am I moving toward j", "is j's face turned toward me".
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        message_dim: int,
        k_neighbors: int = 8,
        edge_extra_dim: int = 0,
    ):
        super().__init__()
        self.k = int(k_neighbors)
        # Edge function phi_e([h_i, h_j, ||x_i - x_j||^2, <invariant contractions>]) -> m_ij
        self.phi_e = MLP(
            feature_dim * 2 + 1 + int(edge_extra_dim), hidden_dim, message_dim, layers=2
        )
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
        vec_cat: Optional[torch.Tensor] = None,
        n_vec: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, 3], h: [B, N, H], mask: [B, N] where 1.0 indicates padding to ignore
        # vec_cat: [B, N, 3 * n_vec] equivariant vector channels, held fixed across layers
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
        k = idx.shape[2]
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, K, 3]
        h_i = h.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, K, H]
        # Compute edge messages
        d_vec = x_i - x_j  # [B, N, K, 3]
        r2 = (d_vec * d_vec).sum(-1, keepdim=True)
        edge_feats = [h_i, h_j, r2]

        if vec_cat is not None and n_vec > 0:
            # Gather neighbour vector channels with the same one-hot trick, then
            # slice per channel. Slicing on a static last axis keeps the graph
            # 4-D throughout -- a stacked [B, N, K, C, 3] layout would need a
            # batch-dependent reshape, which does not survive export.
            v_j_all = torch.matmul(onehot, vec_cat.unsqueeze(1))  # [B, N, K, 3C]
            v_i_list: List[torch.Tensor] = []
            v_j_list: List[torch.Tensor] = []
            for c in range(n_vec):
                lo, hi = 3 * c, 3 * c + 3
                v_i_list.append(
                    vec_cat[..., lo:hi].unsqueeze(2).expand(-1, -1, k, -1)
                )
                v_j_list.append(v_j_all[..., lo:hi])
            # <edge direction, channel> for both endpoints: signed geometry such
            # as height difference (against the gravity channel) and closing
            # speed (against the velocity channel).
            for c in range(n_vec):
                edge_feats.append((d_vec * v_i_list[c]).sum(-1, keepdim=True))
            for c in range(n_vec):
                edge_feats.append((d_vec * v_j_list[c]).sum(-1, keepdim=True))
            # Full channel-vs-channel contraction between the two endpoints:
            # relative orientation and relative motion.
            for ci in range(n_vec):
                for cj in range(n_vec):
                    edge_feats.append(
                        (v_i_list[ci] * v_j_list[cj]).sum(-1, keepdim=True)
                    )

        e_ij = torch.cat(edge_feats, dim=-1)
        m_ij = self.phi_e(e_ij)  # [B, N, K, M]

        # Position update
        scale = self.phi_x(m_ij)  # [B, N, K, 1]
        dx = d_vec * scale
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

    Entity rows are laid out by EGNNSensorComponent.BuildRow in a fixed order:

        pos(3) | rotation quaternion(4)? | linear velocity(3)? | angular velocity(3)? | scalars(rest)

    where the optional blocks mirror the component's m_IncludeRotation /
    m_IncludeLinearVelocity / m_IncludeAngularVelocity toggles and the trailing
    scalars are the type and subtype one-hots.

    attr_mode="equivariant" (default) splits that row by its actual geometric
    type. Quaternions become three equivariant axis vectors, velocities stay
    equivariant vectors, and only genuine invariants (one-hots, channel norms,
    channel-pair dot products) are fed to the node feature channel.

    A constant "up" channel is appended to the vector channels. Full SO(3)
    equivariance is the *wrong* symmetry for a physics environment: gravity
    picks out a vertical axis, and a network that cannot tell up from sideways
    cannot represent uprightness or fall recovery. Supplying gravity as an
    explicit equivariant reference recovers exactly the vertical information
    (v . up is vertical speed, axis_up . up is uprightness, and at edge level
    (x_i - x_j) . up is height difference) while keeping the encoder equivariant
    to yaw and translation -- which is the symmetry the world actually has.

    attr_mode="raw" reproduces the pre-fix behaviour bit for bit: every
    non-positional column is pushed through a single Linear as if it were an
    invariant scalar. Kept so in-flight runs can be resumed without an
    architecture change.
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
        attr_mode: str = "equivariant",
        has_quaternion: bool = False,
        has_linear_velocity: bool = False,
        has_angular_velocity: bool = False,
        up_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    ):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.attr_dim = max(0, int(input_dim) - self.pos_dim)
        self.embedding_size = int(embedding_size)
        self.attr_mode = str(attr_mode)
        if self.attr_mode not in ("equivariant", "raw"):
            raise ValueError(
                f"attr_mode must be 'equivariant' or 'raw', got '{attr_mode}'"
            )
        # Identity constant for kNN self-edge masking and one-hot neighbor gather.
        # Registered as a (non-persistent) buffer so ONNX export emits a plain
        # initializer instead of EyeLike, which Sentis does not support.
        if max_entities is not None:
            self.register_buffer(
                "node_eye", torch.eye(int(max_entities)), persistent=False
            )
        else:
            self.node_eye = None

        edge_extra_dim = 0
        if self.attr_mode == "equivariant":
            self._build_layout(
                input_dim, has_quaternion, has_linear_velocity, has_angular_velocity
            )
            self.register_buffer(
                "up_vec", torch.tensor(list(up_axis), dtype=torch.float32).view(1, 1, 3)
            )
            c = self.n_vec_channels
            # norms + strict-upper-triangle channel dot products
            node_feat_dim = self.scalar_dim + c + (c * (c - 1)) // 2
            # <d, v_i>, <d, v_j>, and the full v_i x v_j contraction
            edge_extra_dim = 2 * c + c * c
        else:
            self.n_vec_channels = 0
            self.scalar_dim = self.attr_dim
            node_feat_dim = self.attr_dim if self.attr_dim > 0 else 1

        self.node_feat_dim = node_feat_dim
        self.edge_extra_dim = edge_extra_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    feature_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    message_dim=message_dim,
                    k_neighbors=k_neighbors,
                    edge_extra_dim=edge_extra_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, self.embedding_size)

    def _build_layout(
        self,
        input_dim: int,
        has_quaternion: bool,
        has_linear_velocity: bool,
        has_angular_velocity: bool,
    ) -> None:
        off = self.pos_dim
        self.quat_slice: Optional[Tuple[int, int]] = None
        self.linvel_slice: Optional[Tuple[int, int]] = None
        self.angvel_slice: Optional[Tuple[int, int]] = None
        if has_quaternion:
            self.quat_slice = (off, off + 4)
            off += 4
        if has_linear_velocity:
            self.linvel_slice = (off, off + 3)
            off += 3
        if has_angular_velocity:
            self.angvel_slice = (off, off + 3)
            off += 3
        if off > int(input_dim):
            raise ValueError(
                f"EGNN entity layout consumes {off} columns but the sensor emits only "
                f"{input_dim}. Check that has_quaternion/has_linear_velocity/"
                f"has_angular_velocity match the Unity EGNNSensorComponent toggles."
            )
        self.scalar_slice = (off, int(input_dim))
        self.scalar_dim = int(input_dim) - off
        # 3 axes per quaternion, 1 per velocity, plus the constant gravity channel
        self.n_vec_channels = (
            (3 if has_quaternion else 0)
            + (1 if has_linear_velocity else 0)
            + (1 if has_angular_velocity else 0)
            + 1
        )

    def layout_description(self) -> str:
        if self.attr_mode != "equivariant":
            return f"attr_mode=raw, attrs={self.attr_dim} (all treated as invariant)"
        parts = [f"pos[0:{self.pos_dim}]"]
        if self.quat_slice is not None:
            parts.append(f"quat[{self.quat_slice[0]}:{self.quat_slice[1]}]->3 axes")
        if self.linvel_slice is not None:
            parts.append(f"linvel[{self.linvel_slice[0]}:{self.linvel_slice[1]}]")
        if self.angvel_slice is not None:
            parts.append(f"angvel[{self.angvel_slice[0]}:{self.angvel_slice[1]}]")
        parts.append("up(const)")
        parts.append(f"scalars[{self.scalar_slice[0]}:{self.scalar_slice[1]}]")
        return (
            f"attr_mode=equivariant, {' | '.join(parts)}, "
            f"vec_channels={self.n_vec_channels}, node_feat_dim={self.node_feat_dim}, "
            f"edge_extra_dim={self.edge_extra_dim}"
        )

    @staticmethod
    def mask_from_entities(entities: torch.Tensor) -> torch.Tensor:
        # Padding rows are all-zeros
        return (torch.sum(entities * entities, dim=-1) < 1e-4).float()

    def _vector_channels(self, entities: torch.Tensor) -> List[torch.Tensor]:
        chans: List[torch.Tensor] = []
        if self.quat_slice is not None:
            lo, hi = self.quat_slice
            right, up, forward = quat_to_axes(entities[..., lo:hi])
            chans.extend([right, up, forward])
        if self.linvel_slice is not None:
            lo, hi = self.linvel_slice
            chans.append(entities[..., lo:hi])
        if self.angvel_slice is not None:
            lo, hi = self.angvel_slice
            chans.append(entities[..., lo:hi])
        # Constant gravity reference, broadcast without an Expand node
        chans.append(entities[..., 0:3] * 0.0 + self.up_vec)
        return chans

    def _node_invariants(
        self, vecs: List[torch.Tensor], scalars: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        if self.scalar_dim > 0:
            feats.append(scalars)
        for v in vecs:
            feats.append(torch.sqrt((v * v).sum(-1, keepdim=True).clamp_min(eps)))
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                feats.append((vecs[i] * vecs[j]).sum(-1, keepdim=True))
        return torch.cat(feats, dim=-1)

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        # entities: [B, N, D]
        b, n, d = entities.shape
        mask = self.mask_from_entities(entities)  # [B, N]
        pos = entities[..., : self.pos_dim]

        if self.attr_mode == "raw":
            if self.attr_dim > 0:
                attrs = entities[..., self.pos_dim :]
            else:
                attrs = torch.ones(
                    b, n, 1, device=entities.device, dtype=entities.dtype
                )
            h = self.input_proj(attrs)
            vec_cat = None
            n_vec = 0
        else:
            vecs = self._vector_channels(entities)
            lo, hi = self.scalar_slice
            h = self.input_proj(self._node_invariants(vecs, entities[..., lo:hi]))
            vec_cat = torch.cat(vecs, dim=-1)  # [B, N, 3C]
            n_vec = self.n_vec_channels

        for layer in self.layers:
            pos, h = layer(pos, h, mask, self.node_eye, vec_cat, n_vec)
        out = self.out_proj(h)
        # Zero out padded nodes for cleanliness
        out = out * (1.0 - mask).unsqueeze(-1)
        return out  # [B, N, E]
