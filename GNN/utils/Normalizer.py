import torch
from torch import tensor, zeros, sum, max, sqrt, float
from torch.nn import Module
import torch.distributed as dist


class Normalizer(Module):
    def __init__(
        self,
        size,
        max_accumulations=10**7,
        epsilon=1e-8,
        ignore_mask=None,
        device=None,
        name=None,
    ):

        """
        Online normalization module

        size: feature dimension
        max_accumulation: maximum number of batches
        epsilon: std cutoff for constant variable
        ignore_mask: binary array indicating whether to normalize an entry (0) or not (1)
        device: pytorch device
        """

        super(Normalizer, self).__init__()

        self.max_accumulations = max_accumulations
        self.epsilon = epsilon

        if ignore_mask is None:
            self.ignore = zeros(size, dtype=bool)
        else:
            assert ignore_mask.shape[0] == size
            self.ignore = torch.tensor(ignore_mask, dtype=bool)

        self.register_buffer("acc_count", tensor(0, dtype=float, device=device))
        self.register_buffer("num_accumulations", tensor(0, dtype=float, device=device))
        self.register_buffer("acc_sum", zeros(size, dtype=float, device=device))
        self.register_buffer("acc_sum_squared", zeros(size, dtype=float, device=device))

        if name is not None:
            self.name = name

    def forward(self, batched_data, accumulate=True):
        """
        Updates mean/standard deviation and normalizes input data

        batched_data: batch of data
        accumulate: if True, update accumulation statistics
        """

        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)

        # Note: Added self.epsilon instead of std[std < self.epsilon] = 1.0
        return (batched_data - self._mean().to(batched_data.device)) / (
            self._std().to(batched_data.device) + self.epsilon
        )

    def inverse(self, normalized_batch_data):
        """
        Unnormalizes input data
        """

        return normalized_batch_data * self._std().to(
            normalized_batch_data.device
        ) + self._mean().to(normalized_batch_data.device)

    def _accumulate(self, batched_data):
        """
        Accumulates statistics for mean/standard deviation computation
        """
        count = tensor(batched_data.shape[0]).float().to(self.acc_count.device)
        data_sum = sum(batched_data, dim=0).to(self.acc_sum.device)
        squared_data_sum = sum(batched_data**2, dim=0).to(self.acc_sum_squared.device)
        num_acc_incr = torch.tensor(1).to(self.num_accumulations.device)

        # reduce method
        if dist.is_initialized():
            dist.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(data_sum, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(squared_data_sum, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(num_acc_incr, op=torch.distributed.ReduceOp.SUM)
        self.acc_sum += data_sum
        self.acc_sum_squared += squared_data_sum
        self.acc_count += count
        self.num_accumulations += num_acc_incr

    def _mean(self):
        """
        Returns accumulated mean
        """
        safe_count = max(self.acc_count, tensor(1.0).float())
        mean = self.acc_sum / safe_count
        mean[self.ignore] = 0.0

        return mean

    def _std(self):
        """
        Returns accumulated standard deviation
        """
        safe_count = max(self.acc_count, tensor(1.0).float())
        std = sqrt(self.acc_sum_squared / safe_count - self._mean() ** 2)
        std[self.ignore] = 1.0

        return std
