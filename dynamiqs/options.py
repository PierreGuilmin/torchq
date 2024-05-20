from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jax import Array
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter

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
        t0: Initial time. If `None`, defaults to the first time in `tsave`.
        save_extra _(function, optional)_: A function with signature
            `f(Array) -> PyTree` that takes a state as input and returns a PyTree.
            This can be used to save additional arbitrary data during the
            integration.
    """

    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    progress_bar: AbstractProgressMeter | None = TqdmProgressMeter()
    t0: ScalarLike | None = None
    save_extra: callable[[Array], PyTree] | None = None

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        progress_bar: AbstractProgressMeter | None = TqdmProgressMeter(),  # noqa: B008
        t0: ScalarLike | None = None,
        save_extra: callable[[Array], PyTree] | None = None,
    ):
        if progress_bar is None:
            progress_bar = NoProgressMeter()

        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.progress_bar = progress_bar
        self.t0 = t0

        # make `save_extra` a valid Pytree with `Partial`
        if save_extra is not None:
            save_extra = jtu.Partial(save_extra)
        self.save_extra = save_extra

    def __str__(self) -> str:
        return tree_str_inline(self)
