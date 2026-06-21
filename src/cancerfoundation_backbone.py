from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, gene_ids: Tensor) -> Tensor:
        return self.norm(self.embedding(gene_ids))


class ContinuousValueEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_value: int,
    ) -> None:
        super().__init__()
        self.max_value = max_value
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        values = torch.clamp(values, max=self.max_value).unsqueeze(-1)
        return self.net(values)


class Adapter(nn.Module):
    """Residual bottleneck adapter for parameter-efficient finetuning."""

    def __init__(
        self,
        *,
        d_model: int,
        bottleneck_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
        )
        # Start as an exact identity through the residual path. This preserves
        # the pretrained backbone at adapter insertion and lets the adapter
        # contribution grow from zero during fine-tuning.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.net(hidden)


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer with optional residual adapters after attention/FFN."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_adapter: nn.Module | None = None
        self.ff_adapter: nn.Module | None = None

    def add_adapters(
        self,
        *,
        bottleneck_dim: int = 32,
        dropout: float = 0.0,
        after_attention: bool = True,
        after_ff: bool = True,
    ) -> list[nn.Module]:
        if not after_attention and not after_ff:
            raise ValueError("At least one adapter insertion point must be enabled.")
        if self.attn_adapter is not None or self.ff_adapter is not None:
            raise ValueError("Adapters have already been added to this encoder layer.")

        adapters: list[nn.Module] = []
        d_model = int(self.linear2.out_features)
        if after_attention:
            self.attn_adapter = Adapter(
                d_model=d_model,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
            )
            adapters.append(self.attn_adapter)
        if after_ff:
            self.ff_adapter = Adapter(
                d_model=d_model,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
            )
            adapters.append(self.ff_adapter)
        return adapters

    def adapter_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.attn_adapter is not None:
            params.extend(self.attn_adapter.parameters())
        if self.ff_adapter is not None:
            params.extend(self.ff_adapter.parameters())
        return params

    def _self_attention_block(
        self,
        hidden: Tensor,
        src_mask: Tensor | None,
        src_key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        try:
            attn_output = self.self_attn(
                hidden,
                hidden,
                hidden,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
        except TypeError:
            attn_output = self.self_attn(
                hidden,
                hidden,
                hidden,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )[0]
        return self.dropout1(attn_output)

    def _feed_forward_block(self, hidden: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(hidden)))))

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        hidden = src
        if bool(getattr(self, "norm_first", False)):
            hidden = hidden + self._self_attention_block(
                self.norm1(hidden),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            if self.attn_adapter is not None:
                hidden = self.attn_adapter(hidden)
            hidden = hidden + self._feed_forward_block(self.norm2(hidden))
            if self.ff_adapter is not None:
                hidden = self.ff_adapter(hidden)
            return hidden

        hidden = self.norm1(
            hidden
            + self._self_attention_block(
                hidden,
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
        )
        if self.attn_adapter is not None:
            hidden = self.attn_adapter(hidden)
        hidden = self.norm2(hidden + self._feed_forward_block(hidden))
        if self.ff_adapter is not None:
            hidden = self.ff_adapter(hidden)
        return hidden


class CancerFoundationBackbone(nn.Module):
    """CancerFoundation-style encoder: gene embedding + expression-value MLP + TransformerEncoder."""

    def __init__(
        self,
        *,
        num_gene_tokens: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        pad_gene_id: int,
        max_value: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.grad_checkpoint = False
        self.gene_encoder = GeneEncoder(
            num_gene_tokens,
            d_model,
            padding_idx=pad_gene_id,
        )
        self.value_encoder = ContinuousValueEncoder(
            d_model=d_model,
            dropout=dropout,
            max_value=max_value,
        )
        encoder_layer = AdapterTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, nlayers)

    def add_adapters(
        self,
        *,
        bottleneck_dim: int = 32,
        dropout: float = 0.0,
        after_attention: bool = True,
        after_ff: bool = True,
    ) -> nn.ModuleList:
        adapters = nn.ModuleList()
        for layer in self.encoder.layers:
            if not isinstance(layer, AdapterTransformerEncoderLayer):
                raise TypeError(
                    "CancerFoundationBackbone adapters require AdapterTransformerEncoderLayer."
                )
            for adapter in layer.add_adapters(
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                after_attention=after_attention,
                after_ff=after_ff,
            ):
                adapters.append(adapter)
        return adapters

    def adapter_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.encoder.layers:
            if isinstance(layer, AdapterTransformerEncoderLayer):
                params.extend(layer.adapter_parameters())
        return params

    def enable_grad_checkpoint(self) -> None:
        self.grad_checkpoint = True

    def _encode(
        self,
        embeddings: Tensor,
        src_key_padding_mask: Tensor | None,
    ) -> Tensor:
        if not self.grad_checkpoint or not self.training:
            return self.encoder(
                embeddings,
                src_key_padding_mask=src_key_padding_mask,
            )

        hidden = embeddings
        for layer in self.encoder.layers:
            hidden = checkpoint(
                lambda x, current_layer=layer: current_layer(
                    x,
                    src_key_padding_mask=src_key_padding_mask,
                ),
                hidden,
                use_reentrant=False,
            )
        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden

    def forward(
        self,
        gene_ids: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor | None = None,
        return_gene_embeddings: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        gene_embeddings = self.gene_encoder(gene_ids)
        embeddings = gene_embeddings + self.value_encoder(values)
        hidden = self._encode(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )
        if return_gene_embeddings:
            return hidden, gene_embeddings
        return hidden


class ExpressionBinDecoder(nn.Module):
    """Predict a scalar binned expression value, trained with masked MSE."""

    def __init__(
        self,
        *,
        d_model: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden).squeeze(-1)


class ExpressionClsDecoder(nn.Module):
    """Predict masked expression from <cls> hidden state and masked gene identity."""

    def __init__(
        self,
        *,
        d_model: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, cls_hidden: Tensor, gene_embeddings: Tensor) -> Tensor:
        cls_hidden = cls_hidden.unsqueeze(1).expand(-1, gene_embeddings.shape[1], -1)
        return self.net(torch.cat((cls_hidden, gene_embeddings), dim=-1)).squeeze(-1)
