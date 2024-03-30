import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .utils import (
    get_seq_parallel_receive_rank,
    get_seq_parallel_send_rank,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DBLOCK: tl.constexpr,
    NUM_DBLOCK: tl.constexpr,
    EBLOCK: tl.constexpr,
    NUM_EBLOCK: tl.constexpr,
):
    off_d = tl.program_id(0)
    off_e = tl.program_id(1)
    off_bh = tl.program_id(2)
    off_h = off_bh % h
    # get the (b, h) location
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_d * b * h * n * e + off_bh * n * e
    kv_offset = off_bh * d * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK

    kv_d_offset = d_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + d_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + d_offset
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, DBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    array = tl.arange(0, BLOCK).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * array[:, None])
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)

    # load global KV
    KV = tl.load(KV_block_ptr).to(tl.float32)

    kv = tl.zeros([DBLOCK, EBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q = tl.load(Q_block_ptr).to(tl.float32)
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        qkv_none_diag = tl.dot(q, kv) * q_decay + tl.dot(q, KV) * tl.exp(
            -s.to(tl.float32) * (array[:, None] + i * BLOCK)
        )
        # diag
        qk = tl.dot(q, k_trans) * diag_decay
        qkv_diag = tl.dot(qk, v)

        qkv = qkv_none_diag + qkv_diag

        tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)

        Q_block_ptr += BLOCK * d
        K_trans_block_ptr += BLOCK * d
        V_block_ptr += BLOCK * e
        O_block_ptr += BLOCK * e

    KV = tl.exp(-s.to(tl.float32) * n) * KV + kv
    tl.store(KV_block_ptr, KV.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    KV,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    DBLOCK: tl.constexpr,
    NUM_DBLOCK: tl.constexpr,
    EBLOCK: tl.constexpr,
    NUM_EBLOCK: tl.constexpr,
):
    off_d = tl.program_id(0)
    off_e = tl.program_id(1)
    off_bh = tl.program_id(2)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK

    dqk_offset = off_e * b * h * n * d
    dv_offset = off_d * b * h * n * e

    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK
    kv_d_offset = d_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    DQ_block_ptr = (
        DQ
        + qk_offset
        + dqk_offset
        + d_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    K_block_ptr = (
        K
        + qk_offset
        + d_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + e_offset
        + tl.arange(0, BLOCK)[None, :] * e
        + tl.arange(0, EBLOCK)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    KV_trans_block_ptr = (
        KV
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[None, :] * e
        + tl.arange(0, EBLOCK)[:, None]
    )
    DKV_block_ptr = (
        DKV
        + kv_offset
        + kv_d_offset
        + e_offset
        + tl.arange(0, DBLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    # compute block array
    array = tl.arange(0, BLOCK)

    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)

    KV_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    kv_trans = tl.zeros([EBLOCK, DBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q_decay = tl.exp(-s.to(tl.float32) * array[:, None])
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[:, None]))
        do = tl.load(DO_block_ptr).to(tl.float32)
        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)

        dq_none_diag = tl.dot(do, kv_trans) * q_decay + tl.dot(do, KV_trans) * tl.exp(
            -s.to(tl.float32) * (i * BLOCK + array[:, None])
        )

        dqk = tl.dot(do, v_trans) * diag_decay
        dq_diag = tl.dot(dqk, k)

        dq = dq_none_diag + dq_diag

        tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

        DQ_block_ptr += BLOCK * d
        DO_block_ptr += BLOCK * e
        K_block_ptr += BLOCK * d
        V_trans_block_ptr += BLOCK * e

        kv_trans = block_decay * kv_trans + tl.dot(v_trans, k * k_decay)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + d_offset
        + n * d
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, DBLOCK)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + d_offset
        + n * d
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, DBLOCK)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + e_offset
        + n * e
        + tl.arange(0, BLOCK)[None, :] * e
        + tl.arange(0, EBLOCK)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + dqk_offset
        + d_offset
        + n * d
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, DBLOCK)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + dv_offset
        + e_offset
        + n * e
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + e_offset
        + n * e
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, EBLOCK)[None, :]
    )

    DKV = tl.load(DKV_block_ptr)
    dkv = tl.zeros([DBLOCK, EBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        K_block_ptr -= BLOCK * d
        V_trans_block_ptr -= BLOCK * e
        DK_trans_block_ptr -= BLOCK * d
        DV_block_ptr -= BLOCK * e
        DO_block_ptr -= BLOCK * e
        Q_trans_block_ptr -= BLOCK * d

        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)

        k_decay_trans = tl.exp(-s.to(tl.float32) * (BLOCK - array[None, :]))
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[:, None]))
        q_decay_trans = tl.exp(-s.to(tl.float32) * array[None, :])

        dqk = tl.dot(do, v_trans) * diag_decay
        dk_diag_trans = tl.dot(q_trans, dqk)
        dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans + tl.dot(
            DKV, v_trans.to(DKV.dtype)
        ) * tl.exp(-s.to(tl.float32) * (n - i * BLOCK - array[None, :]))
        dk_trans = dk_none_diag_trans + dk_diag_trans

        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv_diag = tl.dot(qk_trans, do)
        dv_none_diag = tl.dot(k, dkv) * k_decay + tl.dot(k.to(DKV.dtype), DKV) * tl.exp(
            -s.to(tl.float32) * (n - i * BLOCK - array[:, None])
        )
        dv = dv_none_diag + dv_diag

        tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))
        tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))

        dkv = block_decay * dkv + tl.dot(q_trans * q_decay_trans, do)

    DKV = tl.exp(-s.to(tl.float32) * n) * DKV + dkv
    tl.store(DKV_block_ptr, DKV.to(DKV_block_ptr.dtype.element_ty))


def lasp_forward(q, k, v, s, KV):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()
    KV = KV.contiguous()

    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]
    # split over head
    cd = 64
    ce = 64
    d_, e_ = min(triton.next_power_of_2(d), cd), min(triton.next_power_of_2(e), ce)
    nd, ne = d // d_, e // e_
    # right
    o = torch.empty((nd, b, h, n, e), dtype=q.dtype, device=q.device)

    BLOCK = 64

    NUM_BLOCK = q.shape[2] // BLOCK

    grid = (nd, ne, b * h)

    with torch.cuda.device(q.device.index):
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            KV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            DBLOCK=d_,
            NUM_DBLOCK=nd,
            EBLOCK=e_,
            NUM_EBLOCK=ne,
        )

    if nd > 1:
        o = o.sum(0)
    else:
        o.squeeze_()

    return o


def lasp_backward(q, k, v, s, do, KV, DKV):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()
    do = do.contiguous()
    KV = KV.contiguous()
    DKV = DKV.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]
    BLOCK = 32
    NUM_BLOCK = triton.cdiv(n, BLOCK)

    cd = 64
    ce = 64
    d_, e_ = min(triton.next_power_of_2(d), cd), min(triton.next_power_of_2(e), ce)
    nd, ne = d // d_, e // e_

    dq = torch.empty((ne, b, h, n, d), dtype=q.dtype, device=q.device)
    dk = torch.empty((ne, b, h, n, d), dtype=q.dtype, device=q.device)
    dv = torch.empty((nd, b, h, n, e), dtype=q.dtype, device=q.device)

    grid = (
        nd,
        ne,
        b * h,
    )

    with torch.cuda.device(q.device.index):
        _bwd_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            KV,
            DKV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            DBLOCK=d_,
            NUM_DBLOCK=nd,
            EBLOCK=e_,
            NUM_EBLOCK=ne,
        )

    if ne > 1:
        dq = dq.sum(0)
        dk = dk.sum(0)
    else:
        dq.squeeze_(0)
        dk.squeeze_(0)

    if nd > 1:
        dv = dv.sum(0)
    else:
        dv.squeeze_(0)

    return dq, dk, dv


class LaspFuse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, KV, DKV):
        # s: (h, 1, 1)
        b, h, n, d = q.shape
        v.shape[-1]

        KV.zero_()

        group = get_sequence_parallel_group()
        current_idx = get_sequence_parallel_rank()
        send_idx = get_seq_parallel_send_rank()
        recv_idx = get_seq_parallel_receive_rank()

        if current_idx > 0:
            dist.recv(KV, src=send_idx, group=group)

        # need clone, import !!!
        ctx.save_for_backward(q, k, v, s, KV.clone(), DKV)

        o = lasp_forward(q, k, v, s, KV)

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.send(KV, dst=recv_idx, group=group)

        ctx.group = group
        ctx.current_idx = current_idx
        ctx.send_idx = send_idx
        ctx.recv_idx = recv_idx

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, KV, DKV = ctx.saved_tensors
        group = ctx.group
        # forward: 0->1, backward: 1->0
        current_idx = ctx.current_idx
        send_idx = ctx.recv_idx
        recv_idx = ctx.send_idx

        b, h, n, d = q.shape
        v.shape[-1]

        DKV.zero_()

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.recv(DKV, src=send_idx, group=group)

        dq, dk, dv = lasp_backward(q, k, v, s, do, KV, DKV)

        if current_idx > 0:
            dist.send(DKV, dst=recv_idx, group=group)

        return dq, dk, dv, None, None, None, None


lasp_fuse_ = LaspFuse.apply


def lasp_fuse(q, k, v, ed, KV, DKV):
    b, h, n, d = q.shape
    e = v.shape[-1]

    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o = lasp_fuse_(
            q1, k1, v, ed, KV[:, :, s:e].contiguous(), DKV[:, :, s:e].contiguous()
        )
        output = output + o

    return output
