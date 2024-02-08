from __future__ import annotations

from functools import partial
from typing import Any, get_args

import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike

from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Euler, Solver
from ..time_array import TimeArray, _factory_constant
from ..utils.utils import todm
from .mediffrax import MEDopri5, MEEuler


@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def mesolve(
    H: ArrayLike | TimeArray,
    jump_ops: ArrayLike,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: ArrayLike | None = None,
    solver: Solver = Dopri5(),
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    # === convert arguments
    options = {} if options is None else options
    options = Options(**options)

    if isinstance(H, get_args(ArrayLike)):
        H = _factory_constant(H, dtype=options.cdtype)
    Ls = [_factory_constant(L, dtype=options.cdtype) for L in jump_ops]
    y0 = jnp.asarray(psi0, dtype=options.cdtype)
    y0 = todm(y0)
    ts = jnp.asarray(tsave, dtype=options.rdtype)
    E = jnp.asarray(exp_ops, dtype=options.cdtype) if exp_ops is not None else None

    # === select solver class
    solvers = {Euler: MEEuler, Dopri5: MEDopri5}

    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init solver
    solver = solver_class(ts, y0, H, E, solver, gradient, options, Ls)

    # === run solver
    result = solver.run()

    # === return result
    return result
