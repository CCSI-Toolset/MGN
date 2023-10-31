import logging
from pathlib import Path

import torch
import torch.distributed as dist


def init_logger(log_dir, name, rank, world_size, device_ids):
    log_path = Path(log_dir)

    if rank == 0:
        log_path.mkdir(parents=True, exist_ok=True)

    if world_size > 2:
        torch.distributed.barrier(device_ids=device_ids)

    file_name = f"{name}_{rank}_{world_size}.log"
    logging.basicConfig(filename=log_path / file_name, level=logging.DEBUG)


def _allreduce_fut_sum(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    "Sum the input gradient tensor by allreduce and returns a future."
    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()

    def noop(fut):
        return [fut.value()[0]]

    return fut.then(noop)


def allreduce_hook_sum(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future:
    # See: https://pytorch.org/docs/1.9.0/_modules/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.html#allreduce_hook
    return _allreduce_fut_sum(process_group, bucket.get_tensor())
