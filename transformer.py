# stdlib
import math
import copy

# third party
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg, opt):

    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


class Embedder(nn.Module):
    def __init__(self, n_units_input: int, n_units_hidden: int = 500):
        super().__init__()
        self.n_units_hidden = n_units_hidden
        self.embed = nn.Embedding(n_units_input, n_units_hidden)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(
        self, n_units_hidden: int = 500, max_seq_len: int = 500, dropout: float = 0
    ):
        super().__init__()
        self.n_units_hidden = n_units_hidden
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, n_units_hidden)
        for pos in range(max_seq_len):
            for i in range(0, n_units_hidden, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / n_units_hidden)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / n_units_hidden))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.n_units_hidden)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False).to(DEVICE)
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, n_units_hidden: int = 500, eps: float = 1e-6):
        super().__init__()

        self.size = n_units_hidden

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int = 1, n_units_hidden: int = 500, dropout: float = 0):
        super().__init__()

        self.n_units_hidden = n_units_hidden
        self.d_k = n_units_hidden // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(n_units_hidden, n_units_hidden)
        self.v_linear = nn.Linear(n_units_hidden, n_units_hidden)
        self.k_linear = nn.Linear(n_units_hidden, n_units_hidden)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_units_hidden, n_units_hidden)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N n_heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * n_units_hidden
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate n_heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.n_units_hidden)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(
        self, n_units_input: int, n_units_hidden: int = 500, dropout: float = 0
    ):
        super().__init__()

        # We set d_units_hidden as a default to 500
        self.linear_1 = nn.Linear(n_units_input, n_units_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(n_units_hidden, n_units_input)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_units_hidden: int = 500, n_heads: int = 1, dropout: float = 0):
        super().__init__()
        self.norm_1 = Norm(n_units_hidden)
        self.norm_2 = Norm(n_units_hidden)
        self.attn = MultiHeadAttention(n_heads, n_units_hidden, dropout=dropout)
        self.ff = FeedForward(n_units_hidden, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, n_units_hidden: int = 500, n_heads: int = 1, dropout: float = 0):
        super().__init__()
        self.norm_1 = Norm(n_units_hidden)
        self.norm_2 = Norm(n_units_hidden)
        self.norm_3 = Norm(n_units_hidden)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(n_heads, n_units_hidden, dropout=dropout)
        self.attn_2 = MultiHeadAttention(n_heads, n_units_hidden, dropout=dropout)
        self.ff = FeedForward(n_units_hidden, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(
        self,
        n_units_input: int,
        n_units_hidden: int = 500,
        n_layers_hidden: int = 2,
        n_heads: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.n_layers_hidden = n_layers_hidden
        self.embed = Embedder(n_units_input, n_units_hidden)
        self.pe = PositionalEncoder(n_units_hidden, dropout=dropout)
        self.layers = get_clones(
            EncoderLayer(n_units_hidden, n_heads, dropout), n_layers_hidden
        )
        self.norm = Norm(n_units_hidden)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.n_layers_hidden):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        n_units_input: int,
        n_units_hidden: int = 100,
        n_layers_hidden: int = 2,
        n_heads: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.n_layers_hidden = n_layers_hidden
        self.embed = Embedder(n_units_input, n_units_hidden)
        self.pe = PositionalEncoder(n_units_hidden, dropout=dropout)
        self.layers = get_clones(
            DecoderLayer(n_units_hidden, n_heads, dropout), n_layers_hidden
        )
        self.norm = Norm(n_units_hidden)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers_hidden):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        n_units_input: int,
        n_units_output: int,
        n_units_hidden: int = 500,
        n_layers_hidden: int = 2,
        n_heads: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_units_input,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(
            n_units_output,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.out = nn.Linear(n_units_hidden, n_units_output)

    def forward(
        self,
        src_data: torch.Tensor,
        src_mask: torch.Tensor,
        target_data: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        e_outputs = self.encoder(src_data, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, src_mask)
        output = self.out(d_output)
        return output
