from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Solver
from ..time_array import TimeArray
from ..utils.utils import expect


class AbstractSolver(eqx.Module):
    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseSolver(AbstractSolver):
    ts: Array
    y0: Array
    H: Array | TimeArray
    Es: Array
    solver: Solver
    gradient: Gradient | None
    options: Options

    def save(self, y: Array) -> dict[str, Array]:
        saved = {}
        if self.options.save_states:
            saved['ysave'] = y
        if self.Es is not None and len(self.Es) > 0:
            saved['Esave'] = expect(self.Es, y)
        return saved

    def result(self, saved: dict[str, Array]) -> Result:
        ysave = saved.get('ysave', None)
        Esave = saved.get('Esave', None)
        if Esave is not None:
            Esave = Esave.swapaxes(-1, -2)

        return Result(self.ts, self.solver, self.gradient, self.options, ysave, Esave)


SESolver = BaseSolver


class MESolver(BaseSolver):
    Ls: list[Array | TimeArray]
