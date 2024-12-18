
class InverseSqrtScheduler():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, warmup_updates, warmup_init_lr, lr):
        self._optimizer = optimizer
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.n_steps = 0

        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = lr * warmup_updates**0.5

        # initial learning rate
        self.lr = warmup_init_lr
        if self._optimizer is not None:
            self._optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        "Step with the inner optimizer"
        return self._update_learning_rate()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps < self.warmup_updates:
            self.lr = self.warmup_init_lr + self.n_steps * self.lr_step
        else:
            self.lr = self.decay_factor * self.n_steps**-0.5
        if self._optimizer is not None:
            self._optimizer.param_groups[0]['lr'] = self.lr
        return self.lr

    def get_last_lr(self):
        return [self._optimizer.param_groups[0]['lr']]

    # GENE ADDED
    def state_dict(self):
        return {
            'n_steps': self.n_steps,
            'lr': self.lr,
            'warmup_updates': self.warmup_updates,
            'warmup_init_lr': self.warmup_init_lr,
            'lr_step': self.lr_step,
            'decay_factor': self.decay_factor
        }

    def load_state_dict(self, state_dict):
        self.n_steps = state_dict['n_steps']
        self.lr = state_dict['lr']
        self.warmup_updates = state_dict['warmup_updates']
        self.warmup_init_lr = state_dict['warmup_init_lr']
        self.lr_step = state_dict['lr_step']
        self.decay_factor = state_dict['decay_factor']
        # Make sure to update the optimizer's lr as well:
        if self._optimizer is not None:
            self._optimizer.param_groups[0]['lr'] = self.lr


# @dataclass
# class InverseSquareRootLRScheduleConfig(FairseqDataclass):
#     warmup_updates: int = field(
#         default=4000,
#         metadata={"help": "warmup the learning rate linearly for the first N updates"},
#     )
#     warmup_init_lr: float = field(
#         default=-1,
#         metadata={
#             "help": "initial learning rate during warmup phase; default is cfg.lr"
#         },
#     )
#     lr: List[float] = II("optimization.lr")


# @register_lr_scheduler("inverse_sqrt", dataclass=InverseSquareRootLRScheduleConfig)
# class InverseSquareRootSchedule(FairseqLRScheduler):
#     """Decay the LR based on the inverse square root of the update number.

#     We also support a warmup phase where we linearly increase the learning rate
#     from some initial learning rate (``--warmup-init-lr``) until the configured
#     learning rate (``--lr``). Thereafter we decay proportional to the number of
#     updates, with a decay factor set to align with the configured learning rate.

#     During warmup::

#       lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
#       lr = lrs[update_num]

#     After warmup::

#       decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
#       lr = decay_factor / sqrt(update_num)
#     """

#     def __init__(self, cfg: InverseSquareRootLRScheduleConfig, optimizer):
#         super().__init__(cfg, optimizer)
#         if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
#             raise ValueError(
#                 "Cannot use a fixed learning rate schedule with inverse_sqrt."
#                 " Consider --lr-scheduler=fixed instead."
#             )
#         warmup_end_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
#         if cfg.warmup_init_lr < 0:
#             cfg.warmup_init_lr = 0 if cfg.warmup_updates > 0 else warmup_end_lr

#         # linearly warmup for the first cfg.warmup_updates
#         self.lr_step = (warmup_end_lr - cfg.warmup_init_lr) / cfg.warmup_updates

#         # then, decay prop. to the inverse square root of the update number
#         self.decay_factor = warmup_end_lr * cfg.warmup_updates**0.5

#         # initial learning rate
#         self.lr = cfg.warmup_init_lr
#         self.optimizer.set_lr(self.lr)

#     def step(self, epoch, val_loss=None):
#         """Update the learning rate at the end of the given epoch."""
#         super().step(epoch, val_loss)
#         # we don't change the learning rate at epoch boundaries
#         return self.optimizer.get_lr()

#     def step_update(self, num_updates):
#         """Update the learning rate after each update."""
#         if num_updates < self.cfg.warmup_updates:
#             self.lr = self.cfg.warmup_init_lr + num_updates * self.lr_step
#         else:
#             self.lr = self.decay_factor * num_updates**-0.5
#         self.optimizer.set_lr(self.lr)
#         return self.lr