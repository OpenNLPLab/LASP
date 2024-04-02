import argparse

import torch
import torch.distributed as dist
from einops import rearrange

from lasp import (
    lasp_cache,
    lasp_fuse,
    lasp_fuse_parallel,
    lasp_naive,
    lightning_attn,
)
from lasp.utils import (
    build_slope_tensor,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    initialize_lasp,
)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"mean value: {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"mean value: {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


def split_data(x):
    # x: b, h, n, d
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    dp_size = get_data_parallel_world_size(sp_size > 1)
    dp_rank = get_data_parallel_rank(sp_size > 1)

    # split over batch
    x = rearrange(x, "(b g) ... -> g b ... ", g=dp_size)[dp_rank]
    # split over sequence
    x = rearrange(x, "b h (g n) d -> b h g n d", g=sp_size)[:, :, sp_rank]

    return x.detach().clone()


def test(dp_size):
    """
    As an example, assume we have 1 node with 8 GPUs and the ranks are {0, 1, 2, 3, 4, 5, 6, 7}. For data parallel size = 2 and sequence parallel size = 4, the DP and SP communication groups will be:

    4 data_parallel groups (with global rank indices):
    (0, 4), (1, 5), (2, 6), (3, 7)

    2 sequence paralell groups (with global rank indices):
    (0, 1, 2, 3), (4, 5, 6, 7)

    In summary, the group maping (with their own rank indices) is as follows:
    Global ranks:             0, 1, 2, 3, 4, 5, 6, 7
    Data parallel ranks:      0, 0, 0, 0, 1, 1, 1, 1
    Sequence parallel ranks:  0, 1, 2, 3, 0, 1, 2, 3

    In the following example, we initialize data loading on global rank 0, then broadcast data chunks to other ranks. Each GPU gets their own data chunk according to the data parallel rank and sequence parallel rank.
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    sp_size = world_size // dp_size
    initialize_lasp(dp_size, sp_size)

    name_2_fn_dict = {
        "naive": lasp_naive,
        "cache": lasp_cache,
        "fuse": lasp_fuse,
        "fuse_parallel": lasp_fuse_parallel,
    }

    b, n, h, d, e = world_size * 2, 2048, 12, 128, 64

    assert (
        n % sp_size == 0
    ), f"Sequence length {n} must be devided by sequence prallel size {sp_size}"
    b_local = b // dp_size
    n_local = n // sp_size

    # broadcast data on rank 0, then split along batch and sequence dim
    q = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, n, e, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(b, h, n, e, device=device, dtype=dtype, requires_grad=True)
    s = build_slope_tensor(h).to(device).to(torch.float32)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(do, src=0)

    get_sequence_parallel_rank()
    qi, ki, vi, doi = map(split_data, [q, k, v, do])

    qi.requires_grad = True
    ki.requires_grad = True
    vi.requires_grad = True

    dist.barrier()

    o = lightning_attn(q, k, v, s)
    o.backward(do)
    dq = q.grad
    dk = k.grad
    dv = v.grad

    oi_ref, dq_ref, dk_ref, dv_ref = map(split_data, [o, dq, dk, dv])

    for name in name_2_fn_dict:
        qi.grad = None
        ki.grad = None
        vi.grad = None

        f = name_2_fn_dict[name]
        if rank == 0:
            print("\n")
            print(
                f"Test lasp_{name} on world size {world_size} with data_parallel_size {dp_size} and sequence_parallel_size {sp_size}:"
            )

        if rank == 0:
            print("### Forward ###")

        if name == "naive":
            oi = f(qi, ki, vi, s)
        elif name == "cache":
            KV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            DKV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            array = torch.arange(n_local).to(q)
            oi = f(qi, ki, vi, s, array, KV, DKV)
        else:
            KV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            DKV = torch.empty(b_local, h, d, e).to(torch.float32).to(q.device)
            oi = f(qi, ki, vi, s, KV, DKV)

        log("out diff", oi_ref - oi, rank0_only=True)

        dist.barrier()
        if rank == 0:
            print("### Backward ###")

        oi.backward(doi, retain_graph=True)
        dqi = qi.grad.clone()
        dki = ki.grad.clone()
        dvi = vi.grad.clone()

        log("dq diff", dq_ref - dqi, rank0_only=True)
        log("dk diff", dk_ref - dki, rank0_only=True)
        log("dv diff", dv_ref - dvi, rank0_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", help="data parallel size", type=int)
    args = parser.parse_args()
    dp_size = args.dp_size

    test(dp_size)
