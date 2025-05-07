import jax.numpy as jnp
from flax import nnx


class Sequential(nnx.Module):
    def __init__(self, modules: list[nnx.Module]):
        self.modules = modules

    def __call__(self, x: jnp.ndarray):
        for module in self.modules:
            x = module(x)
        return x

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx: int):
        return self.modules[idx]


class ModuleList(nnx.Module):
    def __init__(self, modules: list[nnx.Module]):
        self.modules = modules

    def __call__(self, idx: int, x: jnp.ndarray):
        return self.modules[idx](x)

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx: int):
        return self.modules[idx]
