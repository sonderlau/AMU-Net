import torch
import torch.nn as nn


class TimeInfoEmbedding(nn.Module):
    """Produce a shared time-conditioning vector for AdaMamba modulation."""

    def __init__(
        self,
        embedding_dim: int,
        elapsed_num: int,  # e.g., 2 (sin, cos of time)
        lead_time_max: int,
    ):
        """Initialize the embedding and fusion layers for time information.

        Args:
            embedding_dim: Output dimension of the time embedding.
            elapsed_num: Number of auxiliary elapsed time signals (e.g., sin/cos).
            lead_time_max: Maximum index for the lead-time embedding table.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.elapsed_num = elapsed_num
        self.lead_time_max = lead_time_max

        self.lead_time_embedding = nn.Embedding(
            num_embeddings=lead_time_max + 1,
            embedding_dim=embedding_dim,  # +1 guard against out-of-range indices
        )

        # 2. Elapsed time projection (optional signals such as sin/cos phases)
        if elapsed_num > 0:
            self.elapsed_proj = nn.Linear(elapsed_num, embedding_dim)

        # 3. Fusion MLP - merges lead-time embedding with elapsed projection
        input_dim = embedding_dim * 2 if elapsed_num > 0 else embedding_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),  # Swish activation pairs well with AdaLN
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        lead_time: torch.Tensor,  # (B,) or (B, 1) - int
        elapsed: torch.Tensor,  # (B, elapsed_num) - float
    ) -> torch.Tensor:
        """Return the fused time conditioning vector for the batch.

        Args:
            lead_time: Tensor of lead-time indices with shape (B,) or (B,1).
            elapsed: Auxiliary continuous time signals with shape (B, elapsed_num).

        Returns:
            A tensor of shape (B, embedding_dim) representing global time conditioning.
        """
        
        assert elapsed.shape[1] == self.elapsed_num
        assert lead_time.dim() == 1 or lead_time.dim() == 2

        # 1. Handle lead-time inputs
        if lead_time.dim() == 2:
            lead_time = lead_time.squeeze(1)  # (B, 1) -> (B,)

        # Ensure lead_time stays within defined bounds
        lead_time = torch.clamp(lead_time, max=self.lead_time_max - 1)

        lead_emb = self.lead_time_embedding(lead_time.long())  # (B, D)

        # 2. Handle optional elapsed-time signals
        if self.elapsed_num > 0 and elapsed is not None:
            elapsed_emb = self.elapsed_proj(elapsed)  # (B, D)
            # Concatenate the lead-time and elapsed embeddings
            concat_feat = torch.cat([lead_emb, elapsed_emb], dim=1)  # (B, 2D)
        else:
            concat_feat = lead_emb

        # 3. Apply fusion MLP to combine embeddings
        global_time_cond = self.fusion_mlp(concat_feat)  # (B, D)

        return global_time_cond


class SimpleTimeEmbedding(nn.Module):
    """Create a lightweight time conditioning vector without dedicated embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        elapsed_num: int,  # e.g., 2 (sin, cos of time)
        lead_time_max: int,
    ):
        """Initialize a projection that fuses raw lead-time tokens with elapsed values.

        Args:
            embedding_dim: Output dimension of the fusion MLP.
            elapsed_num: Number of auxiliary elapsed time signals.
            lead_time_max: Maximum allowed lead time value.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.elapsed_num = elapsed_num
        self.lead_time_max = lead_time_max

        # 2. Determine fusion input size based on presence of elapsed signals
        if elapsed_num > 0:
            input_dim = elapsed_num + 1
        else:
            input_dim = 1

        # 3. Fusion MLP - single MLP that mixes lead time and helper features

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),  # Swish activation pairs well with AdaLN
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        lead_time: torch.Tensor,  # (B,) or (B, 1) - int
        elapsed: torch.Tensor,  # (B, elapsed_num) - float
    ) -> torch.Tensor:
        """Compute the fused conditioning vector without learned lead-time embeddings.

        Args:
            lead_time: Lead-time tensor shaped (B,) or (B,1).
            elapsed: Optional continuous elapsed-time signals shaped (B, elapsed_num).

        Returns:
            Tensor of shape (B, embedding_dim) representing the fused conditioning.
        """
        
        assert elapsed.shape[1] == self.elapsed_num
        assert lead_time.dim() == 1 or lead_time.dim() == 2

        # 1. Handle lead-time inputs
        if lead_time.dim() == 1:
            lead_time = lead_time.unsqueeze(1)  # (B,) -> (B,1)

        # Ensure lead_time stays within defined bounds
        lead = torch.clamp(lead_time, max=self.lead_time_max - 1)

        # 2. Handle auxiliary elapsed-time signals (if provided)
        if self.elapsed_num > 0 and elapsed is not None:

            concat_feat = torch.cat([lead, elapsed], dim=1)  # (B, elapsed + 1)
        else:
            concat_feat = lead

        # 3. Produce the fused output through the fusion MLP
        global_time_cond = self.fusion_mlp(concat_feat)  # (B, D)

        return global_time_cond
