import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.decomposition import matrix_product_state   


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random_seed = 1

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.set_printoptions(precision = 8)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class TTLinear(nn.Module):
    def __init__(self, dim, rank):
        super(TTLinear, self).__init__()
        self.input_dim = dim
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)

        query_weight = self.query.weight
        key_weight = self.key.weight
        value_weight = self.value.weight
        weight_matrix = torch.stack([query_weight, key_weight, value_weight])

        weight_matrix_np = weight_matrix.detach().cpu().numpy()
        factors = matrix_product_state(weight_matrix.detach().numpy(), rank)

        self.factors = nn.ParameterList([nn.Parameter(torch.FloatTensor(factor)) for factor in factors])
        
    def forward(self):
        
        return self.factors[0], self.factors[1], self.factors[2]
    
class TTMultiheadAttention(nn.Module):
    def __init__(self, dim, rank, heads, dropout_rate=0.1):
        super(TTMultiheadAttention, self).__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        tt_linear = TTLinear(dim, rank)
        self.ar, self.br, self.cr = tt_linear.forward()
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_dim = dim
        self.rank = rank[2]

    def split_heads(self, x, y):
        x = x.view(x.shape[0], y, -1, x.shape[2], x.shape[3])
        return x

    def forward(self, x):
        df = torch.stack([x, x, x])
        size = df.shape
        df = self.split_heads(df, size[1] // self.rank)

        ar = self.ar
        br = self.br
        cr = self.cr

        attention = torch.matmul(df.permute(3, 1, 2, 4, 0), ar)
        attention = torch.matmul(attention.permute(0, 1, 4, 3, 2), br.permute(0, 2, 1))
        attention = attention / torch.sqrt(torch.tensor(self.head_dim, dtype = torch.float32))
        attention = F.softmax(attention, dim = -1)
        out = torch.matmul(attention, cr.permute(2, 1, 0)).to(device)
        out = out.permute(2, 1, 4, 0, 3)
        out = out.reshape(size)
        out = out.sum(dim = 0).view(x.shape)
            
        return out
    
class c_Transformer(nn.Module):
    def __init__(self, dim, rank, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                TTMultiheadAttention(dim, rank, heads, dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x

class Trans_C(nn.Module):
    def __init__(self, *, dim, rank, depth, heads, mlp_dim, dim_head, dropout, patch_dim, num_vars, learn_pe=True):
        super().__init__()
        self.project = nn.Linear(patch_dim, dim)
        if learn_pe:
            # positional embedding for each variable index (channel)
            self.pos = nn.Parameter(torch.randn(1, num_vars, dim) * 0.02)
        else:
            self.register_buffer("pos", torch.zeros(1, num_vars, dim))
        self.transformer = c_Transformer(dim, rank, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):  # x: [B, nvars, patch_dim]
        x = self.project(x)
        # add positional embedding for available channels
        x = x + self.pos[:, : x.size(1), :]
        return self.transformer(x)  # [B, nvars, dim]
    
class FredformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        c_in: int,
        context_window: int,
        target_window: int,
        d_model: int,
        cf_dim: int,
        cf_depth: int,
        cf_heads: int,
        cf_mlp: int,
        cf_head_dim: int,
        cf_drop: float = 0.1,
        head_dropout: float = 0.1,
        rank: list | tuple | None = None,
    ):
        super().__init__()
        self.rank = rank
        self.freq_len = context_window // 2 + 1          
        self.patch_dim = self.freq_len * 2               
        self.targetwindow = target_window

        # frequency transformer with positional embedding along channel axis
        self.freq_block = Trans_C(
            dim=cf_dim,
            depth=cf_depth,
            heads=cf_heads,
            mlp_dim=cf_mlp,
            dim_head=cf_head_dim,
            dropout=cf_drop,
            rank = rank,
            patch_dim=self.patch_dim,
            num_vars=c_in,
            learn_pe=True,
        )

        self.get_r = nn.Linear(cf_dim, d_model)
        self.get_i = nn.Linear(cf_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, target_window),
        )
        self.combine = nn.Linear(target_window, target_window)

    def forward(self, z):  # z: [B, nvars, seq_len]
        zc = torch.fft.rfft(z)                             
        zc = torch.cat((zc.real, zc.imag), dim=-1)         
        feat = self.freq_block(zc)                         # [B, nvars, cf_dim]
        r = self.head(self.get_r(feat))
        i = self.head(self.get_i(feat))
        sig = torch.fft.irfft(torch.complex(r.float(), i.float()), n=self.targetwindow)
        return self.combine(sig)

class Ours_TT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.core = FredformerBackbone(
            c_in=cfg.enc_in,
            context_window=cfg.seq_len,
            target_window=cfg.pred_len,
            d_model=cfg.d_model,
            cf_dim=cfg.cf_dim,
            cf_depth=cfg.cf_depth,
            cf_heads=cfg.cf_heads,
            cf_mlp=cfg.cf_mlp,
            cf_head_dim=cfg.cf_head_dim,
            cf_drop=cfg.cf_drop,
            head_dropout=cfg.head_dropout,
            rank = getattr(cfg, 'rank', None),
        )

    def forward(self, x):           # [B, T, C]
        return self.core(x.permute(0, 2, 1)).permute(0, 2, 1)