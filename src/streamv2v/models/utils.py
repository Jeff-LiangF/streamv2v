from collections import deque
from typing import Tuple, Callable

from einops import rearrange
import torch
import torch.nn.functional as F

def get_nn_feats(x, y, threshold=0.9):

    if type(x) is deque:
        x = torch.cat(list(x), dim=1)
    if type(y) is deque:
        y = torch.cat(list(y), dim=1)

    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    cosine_similarity = torch.matmul(x_norm, y_norm.transpose(1, 2))

    max_cosine_values, nearest_neighbors_indices = torch.max(cosine_similarity, dim=-1)
    mask = max_cosine_values < threshold
    # print('mask ratio', torch.sum(mask)/x.shape[0]/x.shape[1])
    indices_expanded = nearest_neighbors_indices.unsqueeze(-1).expand(-1, -1, x_norm.size(-1))
    nearest_neighbor_tensor = torch.gather(y, 1, indices_expanded)
    selected_tensor = torch.where(mask.unsqueeze(-1), x, nearest_neighbor_tensor)

    return selected_tensor

def get_nn_latent(x, y, threshold=0.9):

    assert len(x.shape) == 4
    _, c, h, w = x.shape
    x_ = rearrange(x, 'n c h w -> n (h w) c')
    y_ = []
    for i in range(len(y)):
        y_.append(rearrange(y[i], 'n c h w -> n (h w) c'))
    y_ = torch.cat(y_, dim=1)
    x_norm = F.normalize(x_, p=2, dim=-1)
    y_norm = F.normalize(y_, p=2, dim=-1)

    cosine_similarity = torch.matmul(x_norm, y_norm.transpose(1, 2))

    max_cosine_values, nearest_neighbors_indices = torch.max(cosine_similarity, dim=-1)
    mask = max_cosine_values < threshold
    indices_expanded = nearest_neighbors_indices.unsqueeze(-1).expand(-1, -1, x_norm.size(-1))
    nearest_neighbor_tensor = torch.gather(y_, 1, indices_expanded)

    # Use values from x where the cosine similarity is below the threshold
    x_expanded = x_.expand_as(nearest_neighbor_tensor)
    selected_tensor = torch.where(mask.unsqueeze(-1), x_expanded, nearest_neighbor_tensor)

    selected_tensor = rearrange(selected_tensor, 'n (h w) c -> n c h w', h=h, w=w, c=c)

    return selected_tensor


def random_bipartite_soft_matching(
    metric: torch.Tensor, use_grid: bool = False, ratio: float = 0.5
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by a ratio of ratio/2.
    """

    with torch.no_grad():
        B, N, _ = metric.shape
        if use_grid:
            assert ratio == 0.5
            sample = torch.randint(2, size=(B, N//2, 1), device=metric.device)
            sample_alternate = 1 - sample
            grid = torch.arange(0, N, 2).view(1, N//2, 1).to(device=metric.device)
            grid = grid.repeat(4, 1, 1)
            rand_idx = torch.cat([sample + grid, sample_alternate + grid], dim = 1)
        else:
            rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)
        r = int(ratio * N)
        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]
        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge_kv_out(keys: torch.Tensor, values: torch.Tensor, outputs: torch.Tensor, mode="mean") -> torch.Tensor:
        src_keys, dst_keys = split(keys)
        C_keys = src_keys.shape[-1]
        dst_keys = dst_keys.scatter_reduce(-2, dst_idx.expand(B, r, C_keys), src_keys, reduce=mode)

        src_values, dst_values = split(values)
        C_values = src_values.shape[-1]
        dst_values = dst_values.scatter_reduce(-2, dst_idx.expand(B, r, C_values), src_values, reduce=mode)

        src_outputs, dst_outputs = split(outputs)
        C_outputs = src_outputs.shape[-1]
        dst_outputs = dst_outputs.scatter_reduce(-2, dst_idx.expand(B, r, C_outputs), src_outputs, reduce=mode)

        return dst_keys, dst_values, dst_outputs

    def merge_kv(keys: torch.Tensor, values: torch.Tensor, mode="mean") -> torch.Tensor:
        src_keys, dst_keys = split(keys)
        C_keys = src_keys.shape[-1]
        dst_keys = dst_keys.scatter_reduce(-2, dst_idx.expand(B, r, C_keys), src_keys, reduce=mode)

        src_values, dst_values = split(values)
        C_values = src_values.shape[-1]
        dst_values = dst_values.scatter_reduce(-2, dst_idx.expand(B, r, C_values), src_values, reduce=mode)

        return dst_keys, dst_values

    def merge_out(outputs: torch.Tensor, mode="mean") -> torch.Tensor:
        src_outputs, dst_outputs = split(outputs)
        C_outputs = src_outputs.shape[-1]
        dst_outputs = dst_outputs.scatter_reduce(-2, dst_idx.expand(B, r, C_outputs), src_outputs, reduce=mode)

        return dst_outputs
        
    return merge_kv_out, merge_kv, merge_out