"""Communication manager for LASP."""

import operator
from functools import reduce

import torch
import torch.distributed as dist

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# Sequence parallel group, world size, rank
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def get_seq_parallel_send_rank():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return (rank - 1 + world_size) % world_size


def get_seq_parallel_receive_rank():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return (rank + 1 + world_size) % world_size


def initialize_lasp(
    data_parallel_size: int = 1,
    sequence_parallel_size: int = 1,
) -> None:
    """Initialize parallel groups."""
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    enable_linear_attention_sequence_parallel = sequence_parallel_size > 1
    if enable_linear_attention_sequence_parallel:
        if world_size % sequence_parallel_size != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size})"
            )

    sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

    num_data_parallel_groups: int = world_size // data_parallel_size
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = (
        world_size // sequence_parallel_size // data_parallel_size
    )

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    start = 0
    end = world_size
    for i in range(sequence_parallel_size):
        # rank: 0, 1, 2, 3, 4, 5, 6, 7, sp: 4
        # groups: 0, 4
        #         1, 5
        #         2, 6
        #         3, 7
        ranks = range(start + i, end, sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    global _SEQUENCE_DATA_PARALLEL_GROUP
    start = 0
    end = world_size
    ranks = range(start, end)
    group = torch.distributed.new_group(ranks)

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert (
        _SEQUENCE_PARALLEL_GROUP is None
    ), "sequence parallel group is already initialized"

    for i in range(data_parallel_size):
        # rank: 0, 1, 2, 3, 4, 5, 6, 7, dp: 2
        # groups: 0, 1, 2, 3
        #         4, 5 ,6 ,7
        start = i * sequence_parallel_size
        end = start + sequence_parallel_size
        ranks = range(start, end)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    if _SEQUENCE_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert (
        _SEQUENCE_PARALLEL_GROUP is not None
    ), "sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP


def get_data_parallel_group(with_sequence_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_sequence_parallel:
        assert (
            _SEQUENCE_DATA_PARALLEL_GROUP is not None
        ), "sequence data parallel group is not initialized"
        return _SEQUENCE_DATA_PARALLEL_GROUP
    else:
        assert (
            _DATA_PARALLEL_GROUP is not None
        ), "data parallel group is not initialized"
        return _DATA_PARALLEL_GROUP


def set_sequence_parallel_world_size(world_size):
    """Set the sequence  parallel size"""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    _SEQUENCE_PARALLEL_WORLD_SIZE = world_size


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())


def set_sequence_parallel_rank(rank):
    """Set sequence parallel rank."""
    global _SEQUENCE_PARALLEL_RANK
    _SEQUENCE_PARALLEL_RANK = rank


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_sequence_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    assert (
        _DATA_PARALLEL_GLOBAL_RANKS is not None
    ), "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_world_size(with_sequence_parallel=False):
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(
        group=get_data_parallel_group(with_sequence_parallel=False)
    )


def get_data_parallel_rank(with_sequence_parallel=False):
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(
        group=get_data_parallel_group(with_sequence_parallel=False)
    )


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_parallel_groups():
    """Set the groups to none."""
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def get_rank(group):
    return dist.get_rank(group=group)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
