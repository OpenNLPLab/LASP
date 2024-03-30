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
    BLOCK_MODEL: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    # get the (b, h) location
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    e_offset = off_e * BLOCK_MODEL

    # tl.device_print("aaa", BLOCK_MODEL)

    Q_block_ptr = (
        Q + qk_offset + tl.arange(0, BLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K + qk_offset + tl.arange(0, BLOCK)[None, :] * d + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, BLOCK_MODEL)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + e_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, BLOCK_MODEL)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + e_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, BLOCK_MODEL)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    array = tl.arange(0, BLOCK).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * array[:, None])
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - array[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)

    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q = tl.load(Q_block_ptr).to(tl.float32)
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        qkv_none_diag = tl.dot(q, kv) * q_decay
        qk = tl.dot(q, k_trans) * diag_decay
        qkv_diag = tl.dot(qk, v)

        qkv = qkv_none_diag + qkv_diag

        tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)

        Q_block_ptr += BLOCK * d
        K_trans_block_ptr += BLOCK * d
        V_block_ptr += BLOCK * e
        O_block_ptr += BLOCK * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_diag_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h

    #####
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, BLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = (
        DQ
        + qk_offset
        + qk_block_offset
        + tl.arange(0, BLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + qk_block_offset
        + tl.arange(0, BLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + v_block_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + tl.arange(0, BLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    array = tl.arange(0, BLOCK).to(tl.float32)
    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)

    k = tl.load(K_block_ptr).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    do = tl.load(DO_block_ptr).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
    # diag
    dqk = tl.dot(do, v_trans) * diag_decay
    dq_diag = tl.dot(dqk, k)

    dq = dq_diag

    dk_diag_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans) * diag_decay_trans
    dv_diag = tl.dot(qk_trans, do)

    dk_trans = dk_diag_trans
    dv = dv_diag

    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))
    tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


##### all split
@triton.jit
def _bwd_none_diag_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    DQ_block_ptr = (
        DQ
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    DKV_block_ptr = (
        DKV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, e)[None, :]
    )

    # compute block array
    c_array = tl.arange(0, CBLOCK)

    kv_trans = tl.zeros([e, d], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        for j in range(NUM_CBLOCK):
            q_decay = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[:, None]))
            do = tl.load(DO_block_ptr).to(tl.float32)
            dq_none_diag = tl.dot(do, kv_trans) * q_decay
            dq = dq_none_diag + tl.load(DQ_block_ptr)
            tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e

        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
            k = tl.load(K_block_ptr).to(tl.float32)
            k_decay = tl.exp(
                -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
            )
            kv_trans_current += tl.dot(v_trans, k * k_decay)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e

        kv_trans = block_decay * kv_trans + kv_trans_current

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + n * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + n * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + n * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + qk_block_offset
        + n * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + v_block_offset
        + n * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + n * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    dkv = tl.zeros([d, e], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e

            k = tl.load(K_block_ptr).to(tl.float32)
            v_trans = tl.load(V_trans_block_ptr).to(tl.float32)

            k_decay_trans = tl.exp(
                -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :]))
            )
            k_decay = tl.exp(
                -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
            )
            dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans
            dv_none_diag = tl.dot(k, dkv) * k_decay

            dk_trans = dk_none_diag_trans + tl.load(DK_trans_block_ptr)
            dv = dv_none_diag + tl.load(DV_block_ptr)

            tl.store(
                DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty)
            )
            tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))

        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            do = tl.load(DO_block_ptr).to(tl.float32)
            q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
            q_decay_trans = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[None, :]))
            dkv_current += tl.dot(q_trans * q_decay_trans, do)

        dkv = block_decay * dkv + dkv_current
    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


def lasp_forward(q, k, v, s, kv):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()
    kv = kv.contiguous()

    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]
    # right
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

    BLOCK = 64
    NUM_BLOCK = q.shape[2] // BLOCK

    BLOCK_MODEL = 32

    grid = (b * h, e // BLOCK_MODEL)

    with torch.cuda.device(q.device.index):
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            BLOCK_MODEL=BLOCK_MODEL,
        )

    return o


def lasp_backward(q, k, v, s, do):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    do = do.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    b, h, n, d = q.shape
    e = v.shape[-1]
    BLOCK = 32
    NUM_BLOCK = triton.cdiv(n, BLOCK)

    CBLOCK = 16

    assert BLOCK % CBLOCK == 0
    NUM_CBLOCK = BLOCK // CBLOCK

    dkv = torch.empty((b, h, d, e), dtype=q.dtype, device=q.device)

    with torch.cuda.device(q.device.index):
        grid = (b * h, NUM_BLOCK)
        _bwd_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h,)

        _bwd_none_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            dkv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

    return dq, dk, dv, None, dkv


class LaspNaive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s):
        b, h, n, d = q.shape
        e = v.shape[-1]
        array = torch.arange(n).to(q)
        q_decay = torch.exp(-s[None, :].to(torch.float32) * array.reshape(1, 1, -1, 1))
        block_decay = torch.exp(-s[None, :].to(torch.float32) * n)

        KV = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        group = get_sequence_parallel_group()
        current_idx = get_sequence_parallel_rank()
        send_idx = get_seq_parallel_send_rank()
        recv_idx = get_seq_parallel_receive_rank()

        if current_idx > 0:
            dist.recv(KV, src=send_idx, group=group)

        kv = torch.empty(b, h, d, e).to(q)
        o = lasp_forward(q, k, v, s, kv).to(torch.float32) + torch.matmul(
            q * q_decay, KV
        )

        ctx.save_for_backward(q, k, v, s, KV)
        KV = block_decay * KV + kv

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.send(KV, dst=recv_idx, group=group)

        ctx.group = group
        ctx.current_idx = current_idx
        ctx.send_idx = send_idx
        ctx.recv_idx = recv_idx

        o = o.to(q.dtype)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, KV = ctx.saved_tensors
        group = ctx.group

        current_idx = ctx.current_idx
        send_idx = ctx.recv_idx
        recv_idx = ctx.send_idx

        b, h, n, d = q.shape
        e = v.shape[-1]

        array = torch.arange(n).to(do)

        q_decay = torch.exp(-s[None, :].to(torch.float32) * array.reshape(-1, 1))
        k_decay = torch.exp(-s[None, :].to(torch.float32) * (n - array.reshape(-1, 1)))
        block_decay = torch.exp(-s[None, :].to(torch.float32) * n)

        DKV = torch.zeros(b, h, d, e).to(torch.float32).to(do.device)

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.recv(DKV, src=send_idx, group=group)

        dq, dk, dv, _, dkv = lasp_backward(q, k, v, s, do)

        dq = (
            dq.to(torch.float32)
            + torch.matmul(do.to(KV.dtype), KV.transpose(-1, -2)) * q_decay
        )
        dk = (
            dk.to(torch.float32)
            + torch.matmul(v.to(DKV.dtype), DKV.transpose(-1, -2)) * k_decay
        )
        dv = dv.to(torch.float32) + torch.matmul((k * k_decay).to(DKV.dtype), DKV)

        DKV = block_decay * DKV + dkv.to(DKV.dtype)
        if current_idx > 0:
            dist.send(DKV, dst=recv_idx, group=group)

        dq = dq.to(q.dtype)
        dk = dk.to(q.dtype)
        dv = dv.to(q.dtype)

        return dq, dk, dv, None


lasp_naive_ = LaspNaive.apply


def lasp_naive(q, k, v, ed):
    d = q.shape[-1]
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
        o = lasp_naive_(q1, k1, v, ed)
        output = output + o

    return output
