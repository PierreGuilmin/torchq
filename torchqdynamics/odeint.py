import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import torch

from .solver import AdaptativeStep, FixedStep


class ForwardQSolver(ABC):
    @abstractmethod
    def forward(self, t, dt, rho):
        pass


class GradientAlgorithm(Enum):
    AUTOGRAD = 'autograd'  # gradient computed by torch
    NONE = 'none'  # don't compute the gradients
    ADJOINT = 'adjoint'  # compute the gradient using the adjoint method


def odeint(
    qsolver,
    y0,
    t_save: torch.Tensor,
    t_step: torch.Tensor = None,
    exp_ops: List[torch.Tensor] = None,
    save_states: bool = True,
    gradient_algorithm=GradientAlgorithm.AUTOGRAD,
):
    exp_ops = exp_ops or []
    # check arguments
    check_t(t_save)
    t_step is None or check_t(t_step)

    # dispatch to appropriate odeint subroutine
    params = (qsolver, y0, t_save, t_step, exp_ops, save_states)
    if gradient_algorithm == GradientAlgorithm.NONE:
        return _odeint_inplace(*params)
    elif gradient_algorithm == GradientAlgorithm.AUTOGRAD:
        return _odeint_main(*params)
    elif gradient_algorithm == GradientAlgorithm.ADJOINT:
        return _odeint_adjoint(*params)
    else:
        raise ValueError(f'Gradient algorithm {gradient_algorithm} not defined')


def _odeint_main(qsolver, y0, t_save, t_step, exp_ops, save_states):
    if isinstance(qsolver.options, FixedStep):
        return _fixed_odeint(qsolver, y0, t_save, t_step, exp_ops, save_states)
    elif isinstance(qsolver.options, AdaptativeStep):
        warnings.warn(
            't_step argument is not taken into account for adaptative step solvers'
        )
        return _adaptive_odeint(qsolver, y0, t_save, exp_ops, save_states)


# For now we use *args and **kwargs for helper methods that are not implemented to ease the potential API
# changes that could occur later. When a method is implemented the methods should take the same arguments
# as all others
def _odeint_inplace(*args, **kwargs):
    # TODO: Simple solution for now so torch does not store gradients. This
    #       is probably slower than a genuine in-place solver.
    with torch.no_grad():
        return _odeint_main(*args, **kwargs)


def _odeint_adjoint(*_args, **_kwargs):
    raise NotImplementedError


def _adaptive_odeint(*_args, **_kwargs):
    raise NotImplementedError


def _fixed_odeint(qsolver, y0, t_save, t_step, exp_ops, save_states):
    # Initialize save tensor
    y_save = None
    if save_states:
        y_save = torch.zeros(len(t_save), *y0.shape).to(y0)

    if len(exp_ops) > 0:
        exp_save = torch.zeros(len(t_save), len(exp_ops)).to(
            device=y0.get_device(), dtype=torch.float
        )
    else:
        exp_save = None

    # Save first step
    save_counter = 0
    if t_save[0] == 0.0:
        if save_states:
            y_save[0] = y0
        for j, op in enumerate(exp_ops):
            exp_save[save_counter, j] = torch.trace(op @ y0)
        save_counter += 1

    # Run the ODE routine
    y = y0
    for i, t in t_step:
        # Iterate solution
        y = qsolver.forward(t, t[i + 1] - t, y)

        # Save solution
        if t >= t_save[save_counter]:
            if save_states:
                y_save[save_counter] = y

            for j, op in enumerate(exp_ops):
                exp_save[save_counter, j] = torch.trace(op @ y)
            save_counter += 1

    return y_save, exp_save


def check_t(t):
    """Check that `t_save` or `t_step` is valid (it must be a non-empty 1D tensor sorted in
    strictly ascending order and containing only positive values)."""
    if t.dim() != 1 or len(t) == 0:
        raise ValueError(
            'Argument `t_save` and `t_step` must be a non-empty 1D torch.Tensor.'
        )
    if not torch.all(torch.diff(t) > 0):
        raise ValueError(
            'Argument `t_save` and `t_step` must be sorted in strictly ascending order.'
        )
    if not torch.all(t >= 0):
        raise ValueError(
            'Argument `t_save` and `t_step` must contain positive values only.'
        )
