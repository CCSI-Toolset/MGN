from torch.optim.lr_scheduler import _LRScheduler
from math import pow


class ExpLR(_LRScheduler):
    """
    Exponential learning rate scheduler
    Based on procedure described in LTS
    If min_lr==0 and decay_steps==1, same as torch.optim.lr_scheduler.ExpLR
    """

    def __init__(
        self, optimizer, decay_steps=1e6, gamma=0.1, min_lr=1e-6, last_epoch=-1
    ):

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.gamma = gamma
        self.decay_steps = decay_steps

        super(ExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            min_lr
            + max(base_lr - min_lr, 0)
            * pow(self.gamma, self.last_epoch / self.decay_steps)
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]
