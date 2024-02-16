from __future__ import annotations

import equinox as eqx
from jax import Array

from .gradient import Gradient
from .options import Options
from .solver import Solver

__all__ = ['Result']

def memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def memory_str(x: Array) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.2f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.2f} Mb'
    else:
        return f'{mem / 1024**3:.2f} Gb'


def array_str(x: Array) -> str:
    return f'Array {x.dtype} {tuple(x.shape)} | {memory_str(x)}'


class Result(eqx.Module):
    tsave: Array
    solver: Solver
    gradient: Gradient | None
    options: Options
    ysave: Array
    Esave: Array | None

    def __init__(
        self,
        tsave: Array,
        solver: Solver,
        gradient: Gradient | None,
        options: Options,
        ysave: Array,
        Esave: Array | None,
    ):
        """Result of a quantum equation integration.

        Attributes:
            states: State array, alias of `ysave`.
            expects: Expectation array, alias of `Esave`.
            tsave: Time array.
            solver: Solver used.
            gradient: Gradient used.
            options: Options used.
        """
        self.tsave = tsave
        self.solver = solver
        self.gradient = gradient
        self.options = options
        self.ysave = ysave
        self.Esave = Esave

    @property
    def states(self) -> Array:
        # alias for ysave
        return self.ysave

    @property
    def expects(self) -> Array | None:
        # alias for Esave
        return self.Esave

    def __str__(self) -> str:
        parts = {
            'Solver  ': type(self.solver).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'States  ': array_str(self.states),
            'Expects ': array_str(self.expects) if self.expects is not None else None,
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        parts_str = '\n'.join(f'{k}: {v}' for k, v in parts.items())
        return '==== Result ====\n' + parts_str

    def to_qutip(self) -> Result:
        """Convert the Result arrays to QuTiP `QObj` objects.

        Warning:
            Not implemented yet.
        """
        raise NotImplementedError

    def to_numpy(self) -> Result:
        """Convert the Result arrays to NumPy arrays.

        Warning:
            Not implemented yet.
        """
        raise NotImplementedError
