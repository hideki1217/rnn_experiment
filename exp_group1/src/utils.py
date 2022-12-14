import torch


class _RngScope:
    def __init__(self, seed) -> None:
        self.seed = seed
        self._prev = None
    
    def __enter__(self):
        self._prev = torch.get_rng_state()
        torch.manual_seed(self.seed)
        return self
    
    def __exit__(self, ex_type, ex_value, trace):
        torch.set_rng_state(self._prev)


def rng_scope(seed):
    return _RngScope(seed)