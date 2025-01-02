from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter
from .qarrays.qarray import QArray

__all__ = ['Options']


class Options(eqx.Module):
    """Generic options for the quantum solvers.

    Args:
        save_states: If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        verbose: If `True`, print information about the integration, otherwise
            nothing is printed.
        cartesian_batching: If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        progress_meter: Progress meter indicating how far the solve has progressed.
            Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress meter. Pass
            `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        t0: Initial time. If `None`, defaults to the first time in `tsave`.
        save_extra _(function, optional)_: A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.
        max_jumps: Maximum number of jumps for the monto-carlo solver.
    """

    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter()
    t0: ScalarLike | None = None
    save_extra: callable[[QArray], PyTree] | None = None
    max_jumps: int = 100

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),  # noqa: B008
        t0: ScalarLike | None = None,
        save_extra: callable[[QArray], PyTree] | None = None,
        max_jumps: int = 100,
    ):
        if progress_meter is None:
            progress_meter = NoProgressMeter()

        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.progress_meter = progress_meter
        self.t0 = t0

        # make `save_extra` a valid Pytree with `Partial`
        if save_extra is not None:
            save_extra = jtu.Partial(save_extra)
        self.save_extra = save_extra
        self.max_jumps = max_jumps

    def __str__(self) -> str:
        return tree_str_inline(self)
