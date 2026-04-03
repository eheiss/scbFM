import math
import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat

from functools import partial


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = module._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for layer_idx, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[layer_idx] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})

    return routed_args


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    kv_summary = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', kv_summary, q, D_inv)
    return out

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, output_attentions = False):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = k.softmax(dim = -2)

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)
            
        out = linear_attention(q, k, v)
        if output_attentions:
            v_diag = torch.eye(v.shape[-2]).to(device)
            v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0],v.shape[1],1,1)
            attn_weights = torch.zeros(1, 1, q.shape[2], q.shape[2]).to(device).to(torch.float16)
            for head_dim in range(q.shape[1]):
                attn_weights += torch.abs(linear_attention(
                    q[:, head_dim].to(torch.float16),
                    k[:, head_dim].to(torch.float16),
                    v_diag[:, head_dim].to(torch.float16),
                ))
            attn_weights /= q.shape[1]
            return out, attn_weights
        else:
            return out


# classes

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, output_attentions = False, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if output_attentions:
            attn_weights = []

        for (f, g), (f_args, g_args) in layers_and_args:
            if output_attentions:
                attn_out, layer_attn_weights = f(x, output_attentions = output_attentions, **f_args)
                x = x + attn_out
                attn_weights.append(layer_attn_weights.unsqueeze(0))
            else:
                x = x + f(x, **f_args)

            x = x + g(x, **g_args)

        if output_attentions:
            attn_weights = torch.transpose(torch.cat(attn_weights, dim = 0), 0, 1)
            attn_weights = torch.mean(attn_weights, dim = 1)
            return x, attn_weights

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        nb_features = None,
        dropout = 0.,
        no_projection = False,
        qkv_bias = False
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, no_projection = no_projection)

        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, output_attentions = False, **kwargs):
        h = self.heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            global_mask = mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.)

        if output_attentions:
            out, attn_weights = self.fast_attention(q, k, v, output_attentions)
        else:
            out = self.fast_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        if output_attentions:
            return self.dropout(out), attn_weights
        else:
            return self.dropout(out)


# positional embeddings

class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        gene2vec_weight = np.load('../data/gene2vec_16906.npy')
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


# performer

class Performer(nn.Module):
    def __init__(
        self,
        dim,                                # dimension
        depth,                              # layers
        heads,                              # heads
        dim_head,                           # dim of head
        ff_mult = 4,                        # dim of intermediate features after attention / dim of input features
        nb_features = None,                 # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head ?? what is random feature ??
        feature_redraw_interval = 1000,     # how frequently to redraw the projection matrix, the more frequent, the slower the training
        ff_chunks = 1,                      # chunk feedforward layer, from Reformer
        use_scalenorm = False,              # use scale norm, from 'Transformers without Tears' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        use_rezero = False,                 # use Rezero or not, from 'Rezero is all you need' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        ff_glu = False,                     # use GLU (Gated Linear Units) variant for feedforward
        ff_dropout = 0.,                    # feedforward dropout
        attn_dropout = 0.,                  # post-attention dropout
        no_projection = False,              # ??
        auto_check_redraw = True,           # ??
        qkv_bias = True,                    # ??
    ):
        super().__init__()
        layers = nn.ModuleList([])

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _ in range(depth):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}
        self.net = SequentialSequence(layers, args_route = attn_route_map)

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)

            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, output_attentions = False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()
        return self.net(x, output_attentions = output_attentions, **kwargs)

class PerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,                         # num of tokens
        max_seq_len,                        # max length of sequence
        dim,                                # dim of tokens
        depth,                              # layers
        heads,                              # num of heads
        dim_head = 64,                      # dim of heads
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        use_scalenorm = False,
        use_rezero = False,
        no_projection = False,
        tie_embed = False,                  # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
        g2v_position_emb = True,            # priority: gene2vec, no embedding
        auto_check_redraw = True,
        qkv_bias = False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding()
        else:
            self.pos_emb = torch.zeros_like

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(dim, depth, heads, dim_head, ff_mult, nb_features, feature_redraw_interval, ff_chunks, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, no_projection, auto_check_redraw, qkv_bias)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, output_attentions = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        if output_attentions:
            x.requires_grad_()    # used for attn_map output
        x += self.pos_emb(x)
        x = self.dropout(x)

        if output_attentions:
            x, attn_weights = self.performer(x, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x, attn_weights

            if exists(self.to_out):
                return self.to_out(x), attn_weights

            return (x @ self.token_emb.weight.t()), attn_weights
        else:
            x = self.performer(x, output_attentions = output_attentions, **kwargs)

            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x

            if exists(self.to_out):
                x = self.to_out(x)
                return x

            return x @ self.token_emb.weight.t()
        