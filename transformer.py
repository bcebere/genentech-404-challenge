# third party
import torch
import numpy as np
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from typing import Any, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, n_units_hidden: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.n_units_hidden = n_units_hidden
        self._num_timescales = n_units_hidden // 2

    def forward(self, horizons):
        horizons = horizons.float()

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        timescales = torch.from_numpy(timescales).to(DEVICE)
        horizons = horizons.swapaxes(0,1)
        times = horizons.unsqueeze(2)

        scaled_time = times / timescales[None, None, :]
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x n_units_hidden
        pe = pe.type(torch.FloatTensor)

        return pe.to(DEVICE)


class Transformer(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        n_units_hidden: int = 512,
        n_head: int = 1,
        d_ffn: int = 128,
        dropout: float = 0.1,
        activation="relu",
        n_layers_hidden: int = 2,
    ):
        """
        Args:
            n_units_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            n_units_out: the number of target classes
            n_units_hidden: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.

        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
        """
        super(Transformer, self).__init__()

        self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(n_units_in, n_units_hidden)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(
            n_units_hidden,
            n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
        )
        encoder_norm = nn.LayerNorm(n_units_hidden)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers_hidden, norm=encoder_norm
        )
        self.transpose = Transpose(1, 0)
        self.max = Max(1)
        self.outlinear = nn.Linear(n_units_hidden, n_units_out)
        self.pos_encoder = PositionalEncoding(n_units_hidden=n_units_hidden)

    def forward(self, x, horizons):
        assert len(x) == len(horizons) # same batchsize
        assert x.shape[-1] == horizons.shape[-1] # same seq len

        x = self.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x n_units_hidden
        x = self.relu(x)

        pos = self.pos_encoder(horizons)
        x = x + pos

        x = self.transformer_encoder(x)
        x = self.transpose(
            x
        )  # seq_len x bs x n_units_hidden -> bs x seq_len x n_units_hidden
        x = self.max(x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x


class Permute(nn.Module):
    def __init__(self, *dims: Any) -> None:
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class Max(nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super(Max, self).__init__()
        self.dim, self.keepdim = dim, keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(self.dim, keepdim=self.keepdim)[0]


class Transpose(nn.Module):
    def __init__(self, *dims: Any, contiguous: bool = False) -> None:
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)
