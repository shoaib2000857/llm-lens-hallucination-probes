"""Unmodified legacy architecture definitions only; source SHA256 c76c5a9075b86629f1d1e3c0070011f166bfbb79b6b8c3f9b92e116ba90e9507."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def sinusoidal_1d(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding. Returns [length, dim]."""
    pe  = torch.zeros(length, dim)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

def build_2d_pos_encoding(n_layers: int, t_fixed: int, d_model: int) -> torch.Tensor:
    """
    2D positional encoding that separately encodes layer index and token position.
    Returns [n_layers * t_fixed, d_model].
    Layer gets first d_model//2 dims, position gets last d_model//2 dims.
    """
    half      = d_model // 2
    layer_enc = sinusoidal_1d(n_layers, half)                           # [L, d/2]
    pos_enc   = sinusoidal_1d(t_fixed,  half)                           # [T, d/2]
    layer_enc = layer_enc.unsqueeze(1).expand(-1, t_fixed, -1)          # [L, T, d/2]
    pos_enc   = pos_enc.unsqueeze(0).expand(n_layers, -1, -1)           # [L, T, d/2]
    enc_2d    = torch.cat([layer_enc, pos_enc], dim=-1)                 # [L, T, d]
    return enc_2d.reshape(n_layers * t_fixed, d_model)

class ProbeMLP(nn.Module):
    """
    Mean-pool over both L and T → flat [hidden_dim] → MLP → scalar.
    Dumbest possible baseline — if fancier models can't beat this the
    dataset has issues.
    """

    def __init__(self, hidden_dim: int = 4096, proj_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, D]
        return self.net(x.mean(dim=(1, 2))).squeeze(-1)   # [B]

    @property
    def name(self) -> str:
        return "M0_ProbeMLP"

class LayerWiseMLP(nn.Module):
    """
    Mean-pool over T for each layer separately → [L, D].
    Project each layer independently (shared weights), soft-attention over
    layers, then classify.
    Answers the question: which layers carry the hallucination signal?
    """

    def __init__(
        self,
        n_layers:   int   = 32,
        hidden_dim: int   = 4096,
        proj_dim:   int   = 256,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_attn = nn.Linear(proj_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, D]
        pooled = x.mean(dim=2)                                      # [B, L, D]
        B, L, D = pooled.shape
        proj    = self.layer_proj(pooled.reshape(B * L, D))
        proj    = proj.reshape(B, L, -1)                            # [B, L, proj]
        attn_w  = F.softmax(self.layer_attn(proj), dim=1)           # [B, L, 1]
        weighted = (proj * attn_w).sum(dim=1)                       # [B, proj]
        return self.classifier(weighted).squeeze(-1)                # [B]

    @property
    def name(self) -> str:
        return "M1_LayerWiseMLP"

class CrossLayerTransformer(nn.Module):
    """
    Flattens [L, T] → L*T tokens with 2D positional encoding.
    Prepends a CLS token, runs a standard TransformerEncoder,
    then classifies from CLS output.
    Full attention over all 3072 positions — expensive but principled.
    """

    def __init__(
        self,
        n_layers:   int   = 32,
        t_fixed:    int   = 96,
        hidden_dim: int   = 4096,
        proj_dim:   int   = 256,
        n_heads:    int   = 8,
        tf_layers:  int   = 4,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.seq_len   = n_layers * t_fixed
        self.proj_dim  = proj_dim
        self.input_proj = nn.Linear(hidden_dim, proj_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        pos_enc = build_2d_pos_encoding(n_layers, t_fixed, proj_dim)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))       # [1, L*T, d]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.norm        = nn.LayerNorm(proj_dim)
        self.classifier  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, D]
        B      = x.shape[0]
        tokens = self.input_proj(x.reshape(B, self.seq_len, -1))    # [B, L*T, d]
        tokens = tokens + self.pos_enc
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                    # [B, 1+L*T, d]
        out    = self.transformer(tokens)
        return self.classifier(self.norm(out[:, 0, :])).squeeze(-1) # [B]

    @property
    def name(self) -> str:
        return "M2_CrossLayerTransformer"

class HierarchicalTransformer(nn.Module):
    """
    Two-stage attention:
      Stage 1 (local)  : attend within each layer independently → per-layer repr
      Stage 2 (global) : attend across the 32 layer representations

    More structured inductive bias than M2.
    Potentially more interpretable: stage-2 attention reveals which layers matter.
    """

    def __init__(
        self,
        n_layers:   int   = 32,
        t_fixed:    int   = 96,
        hidden_dim: int   = 4096,
        proj_dim:   int   = 256,
        n_heads:    int   = 8,
        tf_layers:  int   = 2,      # layers per stage (2 + 2 = 4 total ≈ M2)
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.n_layers  = n_layers
        self.t_fixed   = t_fixed
        self.proj_dim  = proj_dim
        self.input_proj = nn.Linear(hidden_dim, proj_dim)

        # ── Stage 1: within-layer ─────────────────────────────────────────────
        self.local_cls = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.local_cls, std=0.02)
        pos_local = sinusoidal_1d(t_fixed, proj_dim)
        self.register_buffer("pos_enc_local", pos_local.unsqueeze(0))  # [1, T, d]
        local_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(local_layer, num_layers=tf_layers)

        # ── Stage 2: cross-layer ──────────────────────────────────────────────
        self.global_cls = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.global_cls, std=0.02)
        pos_global = sinusoidal_1d(n_layers, proj_dim)
        self.register_buffer("pos_enc_global", pos_global.unsqueeze(0)) # [1, L, d]
        global_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(global_layer, num_layers=tf_layers)

        self.norm       = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, D]
        B, L, T, D = x.shape

        # Stage 1 ─ treat (B*L) as independent sequences of T tokens
        x_flat    = x.reshape(B * L, T, D)
        tokens    = self.input_proj(x_flat) + self.pos_enc_local    # [B*L, T, d]
        local_cls = self.local_cls.expand(B * L, -1, -1)
        tokens    = torch.cat([local_cls, tokens], dim=1)           # [B*L, 1+T, d]
        local_out = self.local_transformer(tokens)                  # [B*L, 1+T, d]
        layer_repr = local_out[:, 0, :].reshape(B, L, -1)           # [B, L, d]

        # Stage 2 ─ attend across the 32 layer representations
        layer_repr   = layer_repr + self.pos_enc_global             # add layer-pos
        global_cls   = self.global_cls.expand(B, -1, -1)
        layer_tokens = torch.cat([global_cls, layer_repr], dim=1)   # [B, 1+L, d]
        global_out   = self.global_transformer(layer_tokens)        # [B, 1+L, d]
        summary      = self.norm(global_out[:, 0, :])               # [B, d]

        return self.classifier(summary).squeeze(-1)                 # [B]

    @property
    def name(self) -> str:
        return "M3_HierarchicalTransformer"

class CrossLayerAttentionTransformerV2(nn.Module):
    """
    M4++ — Cross-Layer Attention (improved)

    Improvements:
    - Residual connection in cross-layer attention
    - LayerNorm after attention
    - Multi-query per token (k queries instead of 1)
    - Gated fusion (better than raw attention)
    - Sinusoidal positional encoding (more stable)
    """

    def __init__(
        self,
        n_layers: int = 32,
        t_fixed: int = 96,
        hidden_dim: int = 4096,
        proj_dim: int = 256,
        n_heads: int = 8,
        tf_layers: int = 4,
        dropout: float = 0.3,
        n_queries: int = 4,   # 🔥 NEW (multi-query)
    ):
        super().__init__()

        self.n_layers = n_layers
        self.t_fixed = t_fixed
        self.proj_dim = proj_dim
        self.n_queries = n_queries

        # ── Input projection ────────────────────────────────────
        self.input_proj = nn.Linear(hidden_dim, proj_dim)

        # ── Cross-layer attention ───────────────────────────────
        self.layer_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )

        # 🔥 Multiple learnable queries
        self.token_queries = nn.Parameter(
            torch.randn(1, n_queries, proj_dim)
        )

        # Combine multi-query outputs
        self.query_fusion = nn.Linear(n_queries * proj_dim, proj_dim)

        # Residual + norm
        self.layer_norm = nn.LayerNorm(proj_dim)

        # 🔥 Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.Sigmoid()
        )

        # ── Token transformer ───────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tf_layers
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 🔥 Better positional encoding (sinusoidal)
        self.register_buffer(
            "pos_enc",
            self._build_sinusoidal_pos_encoding(t_fixed, proj_dim),
            persistent=False
        )

        # ── Classifier ─────────────────────────────────────────
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 1),
        )

    def _build_sinusoidal_pos_encoding(self, length, dim):
        pe = torch.zeros(length, dim)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, T, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 32, 96, 4096]
        """

        B, L, T, D = x.shape

        # ── Step 1: project ────────────────────────────────────
        x = self.input_proj(x)  # [B, L, T, d]

        # ── Step 2: cross-layer attention per token ────────────
        x = x.permute(0, 2, 1, 3)  # [B, T, L, d]
        x = x.reshape(B * T, L, self.proj_dim)  # [B*T, L, d]

        # 🔥 Multi-query
        q = self.token_queries.expand(B * T, -1, -1)  # [B*T, Q, d]

        attn_out, _ = self.layer_attn(q, x, x)  # [B*T, Q, d]

        # 🔥 Flatten multi-query
        attn_out = attn_out.reshape(B * T, -1)  # [B*T, Q*d]
        attn_out = self.query_fusion(attn_out)  # [B*T, d]

        # 🔥 Residual (mean of layers)
        residual = x.mean(dim=1)  # [B*T, d]
        attn_out = attn_out + residual

        # 🔥 LayerNorm
        attn_out = self.layer_norm(attn_out)

        # 🔥 Gating
        gate = self.gate(attn_out)
        attn_out = attn_out * gate

        # reshape back
        tokens = attn_out.reshape(B, T, self.proj_dim)  # [B, T, d]

        # ── Step 3: token transformer ─────────────────────────
        tokens = tokens + self.pos_enc

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 1+T, d]

        out = self.transformer(tokens)

        # ── Step 4: classification ────────────────────────────
        cls_out = self.norm(out[:, 0, :])
        return self.classifier(cls_out).squeeze(-1)

    @property
    def name(self) -> str:
        return "M4pp_CrossLayerAttention"
