from __future__ import annotations

from functools import partial
from collections import namedtuple
from typing import Union, Tuple

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F
import copy
import  numpy as np

from beartype import beartype

import einx
from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

import torch.distributed as dist
import sys
sys.path.append('C:/Users/24737/Desktop/fsdownload/ADCM_0806/src/models')



from torch.autograd import Function
# distributed相关函数 
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_same_dim(t):
    t = t.contiguous()
    world_size = dist.get_world_size()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)

def has_only_one_value(t):
    return (t == t[0]).all()

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim = dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim = dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

class AllGather(nn.Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

def split_by_rank(x):
    rank = dist.get_rank()
    out = x[rank]

    if isinstance(x, tuple):
        sizes = tuple(map(lambda t: t.shape[0], x))
    else:
        sizes = (x.shape[1],) * x.shape[0]

    sizes = torch.tensor(sizes, device = out.device, dtype = torch.long)
    return out, sizes


#st_moe

# constants

MIN_EXPERT_CAPACITY = 64

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'outputs_repeat',
    'total_aux_loss',
])

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val

    return default() if callable(default) else default

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor related helper functions

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

# rms normalization

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# expert class
# best performing was ff geglu with multiplicative bias (just after gating)

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias

class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim),
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)

class Experts(Module):
    def __init__(
        self,
        experts,
        is_distributed = None,
        allow_var_seq_len = False # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = ModuleList(experts)

        # distributed related settings

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()

        self.allow_var_seq_len = allow_var_seq_len

        # device tracker, since need to manually move experts not in use to CPU in distributed

        self.register_buffer('dummy', torch.ones(1), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = None
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        # declare some variables

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts
        seq_len = shape[-2]

        # for now naively all gather across batch dimension if distributed, optimize later

        world_size = 1
        rank = 0

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            var_seq_len = not has_only_one_value(seq_sizes)

            assert self.allow_var_seq_len or not var_seq_len, 'number of tokens per expert must be the same - if you want the framework to handle it, set `allow_var_seq_len = True` on `Experts`'

            # if variable sequence length, pad

            if var_seq_len:
                max_seq_size = seq_sizes.amax().item()
                x = pad_dim_to(x, max_seq_size, dim = -2)

            # gather and concat across batches, accounting for variable batch sizes

            x, batch_sizes = self.all_gather(x)
            total_batch_size = batch_sizes.sum().item()

            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # the experts in use on the rank

        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')

            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x, experts_per_rank_sizes = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []

        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad = self.training)

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs, sizes = experts_per_rank_sizes)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs, _ = split_by_rank(outs)

            # account for padded sequence length
            outs = outs[..., :seq_len, :]

        assert outs.shape == shape
        return outs

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class TopNGating(Module):

    @beartype
    def __init__(
        self,
        dim,
        length,
        num_gates,
        eps = 1e-9,
        top_n = 2,
        threshold_train: Union[float, Tuple[float, ...]] = 0.2,
        threshold_eval: Union[float, Tuple[float, ...]] = 0.2,
        capacity_factor_train = 2.5,
        capacity_factor_eval = 2.,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_gates = num_gates
        # self.to_gates = nn.Linear(dim, num_gates, bias = False)
        # 第一个线性层，将输入维度 dim*length 映射到 intermediate_dim1
        self.linear1 = nn.Linear(dim, int(dim / 2), bias=False)
        # 第二个线性层，将 intermediate_dim1 映射到 intermediate_dim2
        self.linear2 = nn.Linear(int(dim / 2), int(dim / 4), bias=False)
        # 第三个线性层，将 intermediate_dim2 映射到 num_gates
        self.linear3 = nn.Linear(int(dim / 4), num_gates, bias=False)
        self.activation = nn.ReLU()

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable = not differentiable_topk,
            fused = differentiable_topk_fused # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        # expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity = MIN_EXPERT_CAPACITY
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        # gate_logits = self.to_gates(x)
        x = self.linear1(x)
        # 经过激活函数
        x = self.activation(x)
        # 经过第二个线性层
        x = self.linear2(x)
        # 经过激活函数
        x = self.activation(x)
        # 经过第三个线性层
        gate_logits = self.linear3(x)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k = top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first

        gates = rearrange(gates, '... k -> k ...')
        gate_indices = rearrange(gate_indices, '... k -> k ...')

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0] # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0., 1.)

        should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min = eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), '... -> ... 1')

        mask_cumsum = cumsum_exclusive(mask, dim = -2) # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum') + prev_expert_count

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum')
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        combine_tensor = einx.multiply(
            'k b n, k b n, k b n e, k b n c -> k b n e c',
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity)
        )

        combine_tensor = reduce(combine_tensor, 'k b n e c -> b n e c', 'sum')

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)            
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss, gate_indices

class MoE(Module):

    @beartype
    def __init__(self,
        dim,
        length = 60,
        num_experts = 16,
        expert_hidden_mult = 2,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-1,
        router_z_loss_coef = 1e-2,
        experts: Union[Module, None] = None,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        is_distributed = None,
        allow_var_seq_len = False
    ):
        super().__init__()
        self.dim = dim
        self.length = length
        self.num_experts = num_experts

        self.gate = TopNGating(
            dim,
            length,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        experts = default(experts, lambda: [Expert(dim = self.dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])
        # self.share_expert = Expert(dim = self.dim, hidden_mult = expert_hidden_mult)
        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )
        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-5)
        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
        self,
        x,
        noise_gates = True,
        noise_mult = 1.,
        rewards=None,
        hidden_states=None,
    ):
        x = self.layer_norm(x)
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss, gate_indices = self.gate(x)
        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)
        expert_outputs = self.layer_norm(expert_outputs)
        if x.shape[0] > 1:
            weighted_balance_loss = balance_loss * self.balance_loss_coef
            weighted_router_z_loss = balance_loss * self.router_z_loss_coef
            # combine the losses
            total_aux_loss = weighted_balance_loss + weighted_router_z_loss
            output_merge = expert_outputs
            output_all_experts = None
        else:
            expert_inputs_repeat = x.unsqueeze(1).repeat(1, self.num_experts, 1, 1)  # (64, 16, 60, 128)
            expert_outputs_repeat = self.experts(expert_inputs_repeat)
            expert_outputs_repeat = self.layer_norm(expert_outputs_repeat)
            output_merge = expert_outputs
            output_all_experts = expert_outputs_repeat
            total_aux_loss = None

        return MixtureOfExpertsReturn(output_merge,output_all_experts, total_aux_loss)

# sparse moe block
# in particular, they found that adding a feedforward before or after greatly stabilized the training and improved results

class SparseMoEBlock(Module):

    @beartype
    def __init__(
        self,
        moe: MoE,
        *,
        add_ff_before = False,
        add_ff_after = True
    ):
        super().__init__()
        dim = moe.dim

        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        self.ff_before = Expert(dim, prenorm = True) if add_ff_before else None
        self.ff_after = Expert(dim, prenorm = True) if add_ff_after else None

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):

        # feedforward before

        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer

        residual = x

        moe_out,moe_outputs_repeat, total_aux_loss, Adv_value = self.moe(self.moe_prenorm(x), noise_gates = noise_gates, noise_mult = noise_mult)

        x = moe_out + residual
        x_repeat = moe_outputs_repeat + residual.unsqueeze(1).repeat(1, 16, 1, 1)

        # feedforward after

        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, total_aux_loss, Adv_value)
