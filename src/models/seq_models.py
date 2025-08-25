import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
sys.path.append('C:/Users/24737/Desktop/fsdownload/ADCM_0806/src/models')
import os
from typing import List, Optional, Tuple
import copy


import json
from dataclasses import dataclass
from typing_extensions import Iterable, NamedTuple, TypeAlias, cast, Union, List, Tuple

from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = Union[str, torch.device, None]


#序列建模相关的模块
# 长期依赖模块
class long_modeling(nn.Module):
    def __init__(self, embed_dim=128, device='cuda',window_size=10,stride=3):
        super(long_modeling, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(self.embed_dim)
        config = Mamba2Config(d_model=self.embed_dim, chunk_size=5, n_layer=3, d_conv=12)
        # self.mamba2 = Mamba2(config, device=self.device)
        self.mamba2_r = Mamba2(config, device=self.device)
        self.mamba2_s = Mamba2(config, device=self.device)
        self.mamba2_a = Mamba2(config, device=self.device)
        self.window_size = window_size
        self.stride = stride
        # self.rtg_conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.window_size, groups=self.embed_dim)
        # self.obs_conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.window_size, groups=self.embed_dim)
        # self.act_conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.window_size, groups=self.embed_dim)
        self.inter_merge_liner = nn.Linear(3 * embed_dim, embed_dim)
    def forward(self,hidden_states):
        batch_size, seq_len, embed_dim = hidden_states.size()
        # Step 1: Apply mamba between patches with causal mask
        padding = (0, 0, 2, 0, 0, 0)
        hidden_states_padding = F.pad(hidden_states, padding, mode='constant', value=0)
        inter_patches = hidden_states_padding.unfold(dimension=1, size=self.stride, step=1)
        inter_patches = inter_patches.reshape(inter_patches.shape[0], inter_patches.shape[1],inter_patches.shape[2]*inter_patches.shape[3])
       
        hidden_states_inter = self.inter_merge_liner(inter_patches)
        # hidden_states_inter = hidden_states_inter.unfold(dimension=1, size=self.stride, step=self.stride).permute(0,3,1,2)
        
        #尝试用三个不同的mamba网络做三种不同的patch序列
        inter_patch_otput_r = self.mamba2_r(hidden_states_inter[:,0::3])[0]
        inter_patch_otput_s = self.mamba2_s(hidden_states_inter[:,1::3])[0]
        inter_patch_otput_a = self.mamba2_a(hidden_states_inter[:,2::3])[0]
        inter_patch_output = torch.stack([inter_patch_otput_r, inter_patch_otput_s, inter_patch_otput_a], dim=1)
        inter_patch_output = inter_patch_output.permute(2,1,0,3)
        inter_patch_output = inter_patch_output.reshape(inter_patch_output.shape[0]*inter_patch_output.shape[1],inter_patch_output.shape[2],inter_patch_output.shape[3]).permute(1,0,2)
        inter_patch_output = self.norm(inter_patch_output)

        #其实用卷积也是可以的，但是效果会差几分（SSM的筛选机制比卷积更适合捕捉长期依赖）
        # padded_tensor = torch.nn.functional.pad(hidden_states_inter, (0, 0, self.window_size - 1, 0)).transpose(1, 2)
        # rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
        # obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
        # act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]
        # conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3), act_conv_tensor.unsqueeze(3)), dim=3)
        # conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1).transpose(1, 2)
        # inter_patch_output = self.norm(conv_tensor)
        return inter_patch_output
    
#短期依赖模块
class short_modeling(nn.Module):
    def __init__(self, embed_dim=128, device='cuda',window_size=3):
        super(short_modeling, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.norm = nn.LayerNorm(self.embed_dim)
        self.rtg_conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.window_size, groups=self.embed_dim)
        self.obs_conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.window_size, groups=self.embed_dim)
        self.act_conv1d = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=self.window_size, groups=self.embed_dim) 
    def forward(self, hidden_states):
        window_size = self.window_size
        padded_tensor = torch.nn.functional.pad(hidden_states, (0, 0, window_size - 1, 0)).transpose(1, 2)
        rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
        obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
        act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

        conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3), act_conv_tensor.unsqueeze(3)), dim=3)
        conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1).transpose(1, 2)
        intra_patch_concat = self.norm(conv_tensor)
        return intra_patch_concat
# 结构性序列建模模块
class PatchAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=1, patch_size=6,device=None):
        self.device = device
        super(PatchAttention, self).__init__()
        self.split_size = embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.scale = False
        self.norm = nn.LayerNorm(self.embed_dim)
        self.norm_2 = nn.LayerNorm(self.embed_dim)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.mstf = MSTF(in_channels=self.embed_dim)
        self.long_modeling_1 = long_modeling(device=self.device,window_size=6,embed_dim=self.embed_dim)
        # self.long_modeling_2 = long_modeling(device=self.device,window_size=6,embed_dim=self.embed_dim)
        # self.long_modeling_3 = long_modeling(device=self.device,window_size=10)
        
        self.short_modeling_1 = short_modeling(device=self.device,embed_dim=self.embed_dim)
        # self.short_modeling_2 = short_modeling(device=self.device,embed_dim=self.embed_dim)
        # self.short_modeling_3 = short_modeling(device=self.device)
    
    def forward(self, hidden_states):
        batch_size, seq_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
        # Step 1: Apply mamba between patches with causal mask
        inter_patch_output = self.long_modeling_1(hidden_states)
        # inter_patch_output = self.long_modeling_2(self.norm(inter_patch_output + hidden_states))
        # inter_patch_output = self.long_modeling_3(inter_patch_output)
        #Step 2: Apply conv in patches with causal mask
        intra_patch_output = self.short_modeling_1(self.norm(inter_patch_output + hidden_states))
        # intra_patch_output = self.short_modeling_3(intra_patch_output , stride=3)
        # Step 3: Combine patch-level and patch-inner representations
        combined_output = self.mstf(inter_patch_output,intra_patch_output)
        combined_output_norm = self.norm_2(combined_output) # Broadcast patch_attn_output to match patch_outputs
        return combined_output_norm
    
#其他网络模块
#mamba2模块(捕捉长期依赖)
@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 3  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 6  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2LMHeadModel(args, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(
        self, input_ids: LongTensor, h: Union[List[InferenceCache], List[None], None] = None
    ) -> Tuple[LongTensor, List[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing `input_ids`
        """
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[Tuple[int, List[InferenceCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: Union[InferenceCache, None] = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache) -> Tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)

#聚合模块
class MSTF(nn.Module):
    def __init__(self, in_channels):
        super(MSTF, self).__init__()
        out_channels = in_channels

        self.project1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )

        self.project2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),)
    def forward(self, x0,x1):
        # (B,T,N,C)
        B,T,C = x0.shape
        x_ = torch.cat([x0,x1],0) # 将多个尺度的输入进行拼接: (B,T,C)--concat--> (M*B,T,C)
        x__ = x_.reshape(-1,T,B,C) # 对其进行reshape,以便后续计算: (M*B,T,C)--reshape-->(M,T,B,C)
        x__ = self.project1(x__.permute(0,3,2,1)).permute(0,3,2,1) # 通过一个线性层学习通道之间的相关性: (M,T,B,C)--permute-->(M,C,B,T)--project1-->(M,C,B,T)--permute-->(M,T,B,C)
        weight = F.softmax(x__, dim=0) # 在M维度上执行softmax,得到每个尺度的权重:(M,T,B,C)
        # 加权和
        x_ = x_.reshape(-1,T,B,C)  # 将输入重塑为与weight相同的shape: (M,B,T,C)-->(M,T,B,C)
        out = (weight * x_).sum(0)  # 每个尺度的权重与对应的输入相乘, 然后将多个尺度的输出相加: (M,T,N*B,C) * (M,T,N*B,C)=(M,T,N*B,C); (M,T,N*B,C)--sum-->(T,N*B,C)
        out = out.reshape(B,T,C) # (T,N*B,C)-->(B,T,N,C)
        return self.project2(out.permute(0,2,1)).permute(0,2,1)

# 行为模仿模型

class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
    
class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]
