import torch

import manager


class _RngScope:
    def __init__(self, seed, device) -> None:
        self.seed = seed
        self.device = device
        self._prev_cpu = None
        self._prev_gpu = None

    def __enter__(self):
        self._prev_cpu = torch.get_rng_state()
        self._prev_gpu = torch.cuda.get_rng_state(device=self.device)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        return self

    def __exit__(self, ex_type, ex_value, trace):
        torch.set_rng_state(self._prev_cpu)
        torch.cuda.set_rng_state(self._prev_gpu, device=self.device)


def rng_scope(seed, device):
    return _RngScope(seed, device)


def torch_fix_seed(seed=42):
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_seeds(num_seed):
    return list(range(num_seed))
