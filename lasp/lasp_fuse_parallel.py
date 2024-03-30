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
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
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
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        k_trans = tl.load(K_trans_block_ptr, mask=kv_index[None, :] < n, other=0.0).to(
            tl.float32
        )
        v = tl.load(V_block_ptr, mask=kv_index[:, None] < n, other=0.0).to(tl.float32)

        qk = tl.dot(q, k_trans) * decay

        qkv += tl.dot(qk, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK

    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # compute block array
    c_array = tl.arange(0, CBLOCK)

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)
        k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :])))

        kv += tl.dot(k_trans * k_decay, v)

        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    K,
    V,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    KV_block_ptr = (
        KV
        + kv_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))

        kv = block_decay * kv + kv_current
        KV_block_ptr += d * e

    # for GKV
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


##### total parallel
@triton.jit
def _fwd_none_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    KV,
    GKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK

    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e + e_offset
    gkv_offset = off_bh * d * e + e_offset

    Q_block_ptr = (
        Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )
    GKV_block_ptr = (
        GKV
        + gkv_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    GKV = tl.load(GKV_block_ptr).to(tl.float32)
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q = tl.load(Q_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (c_offset + c_array[:, None]))
    qkv_none_diag = tl.dot(q, kv) * q_decay + tl.dot(q, GKV) * tl.exp(
        -s.to(tl.float32) * (c_offset + c_array[:, None] + n_offset)
    )
    qkv_diag = tl.load(O_block_ptr).to(tl.float32)

    qkv = qkv_diag + qkv_none_diag

    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))


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
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)

    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    cblock_offset = off_cblock * CBLOCK
    qk_cblock_offset = cblock_offset * d
    v_cblock_offset = cblock_offset * e
    o_cblock_offset = cblock_offset * e

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
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

    do = tl.load(DO_block_ptr).to(tl.float32)
    dq = tl.zeros([CBLOCK, d], dtype=tl.float32)

    i = off_cblock
    do_index = tl.arange(0, CBLOCK) + i * CBLOCK
    for j in range(i + 1):
        k = tl.load(K_block_ptr).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr).to(tl.float32)

        # compute
        v_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = do_index[:, None] - v_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dq += tl.dot(dqk, k)

        K_block_ptr += CBLOCK * d
        V_trans_block_ptr += CBLOCK * e

    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

    # dk
    V_trans_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + v_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )

    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    v_index = tl.arange(0, CBLOCK) + i * CBLOCK
    dk_trans = tl.zeros([d, CBLOCK], dtype=tl.float32)

    # add
    K_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + qk_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + v_block_offset
        + v_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    dv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    k = tl.load(K_block_ptr).to(tl.float32)
    for j in range(i, NUM_CBLOCK):
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        do = tl.load(DO_block_ptr).to(tl.float32)

        do_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = do_index[:, None] - v_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        diag_decay = tl.exp(s_index)

        dqk = tl.dot(do, v_trans) * diag_decay
        dk_trans += tl.dot(q_trans, dqk)

        Q_trans_block_ptr += CBLOCK * d
        DO_block_ptr += CBLOCK * e

        # add
        diag_decay_trans = tl.trans(diag_decay)
        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv += tl.dot(qk_trans, do)

    tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_dkv_parallel(
    Q,
    DO,
    S,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)

    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK

    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    o_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    qk_offset = off_bh * n * d
    o_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    # (CBLOCK, FBLOCK)
    DKV_block_ptr = (
        DKV
        + kv_offset
        + kv_block_offset
        + d_offset * e
        + e_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + o_block_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    c_array = tl.arange(0, CBLOCK)

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    for j in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        q_decay_trans = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[None, :]))
        dkv += tl.dot(q_trans * q_decay_trans, do)

        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d

    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_dkv_reduce(
    Q,
    DO,
    S,
    DKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK

    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + e_offset
        + NUM_BLOCK * d * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

    # compute block array

    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr).to(tl.float32)
        tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))

        dkv = block_decay * dkv + dkv_current

    # store at last pos
    DKV_block_ptr += NUM_BLOCK * d * e
    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


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
    KV,
    DKV,
    GKV,
    GDKV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_de = tl.program_id(2)

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    d_offset = off_de * D_FBLOCK
    e_offset = off_de * E_FBLOCK

    qk_offset = off_bh * n * d + (n_offset + c_offset) * d
    v_offset = off_bh * n * e + (n_offset + c_offset) * e
    o_offset = off_bh * n * e + (n_offset + c_offset) * e

    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e
    kv_trans_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e
    gkv_offset = off_bh * d * e + e_offset
    gkv_trans_offset = off_bh * d * e + e_offset

    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    # dq
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    DQ_block_ptr = (
        DQ
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, D_FBLOCK)[None, :]
    )
    KV_trans_block_ptr = (
        KV
        + kv_trans_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )
    GKV_trans_block_ptr = (
        GKV
        + gkv_trans_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    c_array = tl.arange(0, CBLOCK)
    GKV_trans = tl.load(GKV_trans_block_ptr).to(tl.float32)
    kv_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    q_decay = tl.exp(-s.to(tl.float32) * (c_offset + c_array[:, None]))
    do = tl.load(DO_block_ptr).to(tl.float32)
    dq_none_diag = tl.dot(do, kv_trans) * q_decay + tl.dot(do, GKV_trans) * tl.exp(
        -s.to(tl.float32) * (n_offset + c_offset + c_array[:, None])
    )
    dq = dq_none_diag + tl.load(DQ_block_ptr)
    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))

    # dk
    DK_trans_block_ptr = (
        DK
        + qk_offset
        + d_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DKV_block_ptr = (
        DKV
        + kv_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    GDKV_block_ptr = (
        GDKV
        + gkv_offset
        + d_offset * e
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    GDKV_ = tl.load(GDKV_block_ptr).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    dkv = tl.load(DKV_block_ptr).to(tl.float32)
    k_decay_trans = tl.exp(
        -s.to(tl.float32) * (BLOCK - (off_c * CBLOCK + c_array[None, :]))
    )

    # !!! important !!!
    dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans + tl.dot(
        GDKV_, v_trans.to(GDKV_.dtype)
    ) * tl.exp(-s.to(tl.float32) * (n - n_offset - (c_offset + c_array[None, :])))

    dk_trans = dk_none_diag_trans + tl.load(DK_trans_block_ptr)
    tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.element_ty))

    # dv
    DKV_block_ptr_ = (
        DKV
        + kv_offset
        + e_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    GDKV_block_ptr_ = (
        GDKV
        + gkv_offset
        + e_offset
        + tl.arange(0, d)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + e_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    k_decay = tl.exp(-s.to(tl.float32) * (BLOCK - (off_c * CBLOCK + c_array[:, None])))
    k = tl.load(K_block_ptr).to(tl.float32)

    dkv_ = tl.load(DKV_block_ptr_).to(tl.float32)
    GDKV__ = tl.load(GDKV_block_ptr_).to(tl.float32)
    dv_none_diag = tl.dot(k, dkv_) * k_decay + tl.dot(
        k.to(GDKV__.dtype), GDKV__
    ) * tl.exp(-s.to(tl.float32) * (n - n_offset - (c_offset + c_array[:, None])))
    dv = dv_none_diag + tl.load(DV_block_ptr)
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


def lasp_forward(q, k, v, s, KV, BLOCK=128, CBLOCK=64):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    # shape constraints
    b, h, n, d = q.shape
    e = v.shape[-1]
    # right
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

    NUM_BLOCK = q.shape[2] // BLOCK

    NUM_CBLOCK = BLOCK // CBLOCK

    grid = (b * h * NUM_BLOCK, NUM_CBLOCK)

    with torch.cuda.device(q.device.index):
        _fwd_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
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

    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)

    kv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)

    with torch.cuda.device(q.device.index):
        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _fwd_kv_parallel[grid](
            k,
            v,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _fwd_kv_reduce[grid](
            k,
            v,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            kv,
            KV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )
    block_decay = torch.exp(-s.to(torch.float32) * n)
    KV = block_decay * KV + kv[:, :, -1]

    return o, kv, KV


def lasp_backward(q, k, v, s, do, kv, KV, DKV, BLOCK=128, CBLOCK=64):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    do = do.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    grid = (q.shape[0] * q.shape[1], 1)

    b, h, n, d = q.shape
    e = v.shape[-1]

    # must the same as fwd
    NUM_BLOCK = n // BLOCK

    assert BLOCK % CBLOCK == 0
    NUM_CBLOCK = BLOCK // CBLOCK

    with torch.cuda.device(q.device.index):
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
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

    dkv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32, device=q.device)
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0

    with torch.cuda.device(q.device.index):
        grid = (b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK)
        _bwd_dkv_parallel[grid](
            q,
            do,
            s,
            dkv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_FBLOCK, NUM_FBLOCK)
        _bwd_dkv_reduce[grid](
            q,
            do,
            s,
            dkv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK)
        _bwd_none_diag_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            kv,
            dkv,
            KV,
            DKV,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

    block_decay = torch.exp(-s.to(torch.float32) * n)
    DKV = block_decay * DKV + dkv[:, :, -1]

    return dq, dk, dv, DKV


class LaspFuseParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, KV, DKV):
        # s: (h, 1, 1)
        b, h, n, d = q.shape
        v.shape[-1]

        if n > 128:
            BLOCK = 256
            CBLOCK = 64
        else:
            BLOCK = min(n, 128)
            CBLOCK = min(n, 64)

        KV.zero_()

        group = get_sequence_parallel_group()
        current_idx = get_sequence_parallel_rank()
        send_idx = get_seq_parallel_send_rank()
        recv_idx = get_seq_parallel_receive_rank()

        dist.barrier()
        if current_idx > 0:
            dist.recv(KV, src=send_idx, group=group)

        # need clone, important !!!
        KV_ = KV.clone()

        o, kv, KV = lasp_forward(q, k, v, s, KV, BLOCK, CBLOCK)

        ctx.save_for_backward(q, k, v, s, kv, KV_, DKV)

        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.send(KV, dst=recv_idx, group=group)

        ctx.group = group
        ctx.current_idx = current_idx
        ctx.send_idx = send_idx
        ctx.recv_idx = recv_idx
        ctx.BLOCK = BLOCK
        ctx.CBLOCK = CBLOCK

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s, kv, KV, DKV = ctx.saved_tensors
        BLOCK = ctx.BLOCK
        CBLOCK = ctx.CBLOCK
        group = ctx.group
        # forward: 0->1, backward: 1->0
        current_idx = ctx.current_idx
        send_idx = ctx.recv_idx
        recv_idx = ctx.send_idx

        DKV.zero_()
        if current_idx < get_sequence_parallel_world_size() - 1:
            dist.recv(DKV, src=send_idx, group=group)

        dq, dk, dv, DKV = lasp_backward(q, k, v, s, do, kv, KV, DKV, BLOCK, CBLOCK)

        if current_idx > 0:
            dist.send(DKV, dst=recv_idx, group=group)

        return dq, dk, dv, None, None, None, None


lasp_fuse_parallel_ = LaspFuseParallel.apply


def lasp_fuse_parallel(q, k, v, ed, KV, DKV):
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
        o = lasp_fuse_parallel_(
            q1, k1, v, ed, KV[:, :, s:e].contiguous(), DKV[:, :, s:e].contiguous()
        )
        output = output + o

    return output
