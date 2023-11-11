from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import FunctionCtx
from tqdm.std import TqdmWarning

from ..solver import AdjointSolver
from ..utils.utils import tqdm


class AdjointAutograd(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        solver: AdjointSolver,
        y0: Tensor,
        *params: tuple[nn.Parameter, ...],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.solver = solver

        # integrate the ODE forward without storing the graph of operations
        solver.run_nograd()

        # save results and model parameters
        ctx.save_for_backward(solver.ysave)

        # returning `ysave` is required for custom backward functions
        return solver.ysave, solver.exp_save

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_y: Tensor) -> tuple[None, Tensor, Tensor]:
        """Backward pass of the ODE integrator.

        An augmented ODE is integrated backwards starting from the final state computed
        during the forward pass. Integration is done in multiple sequential runs
        between every checkpoint of the forward pass, as defined by `tstop`. This
        helps with the stability of the backward integration.

        Throughout this function, `y` is the state, `a = dL/dy` is the adjoint state,
        and `g = dL/dp` is the gradient w.r.t. the parameters, where `L` is the loss
        function and `p` the parameters.
        """
        # unpack context
        solver = ctx.solver
        ysave = ctx.saved_tensors[0]

        # locally disable gradient computation
        with torch.no_grad():
            # initialize state, adjoint and gradients
            if solver.options.save_states:
                y = ysave[..., -1, :, :]
                a = grad_y[0][..., -1, :, :]
            else:
                y = ysave[..., :, :]
                a = grad_y[0][..., :, :]
            if len(solver.exp_ops) > 0:
                a += (grad_y[1][..., :, -1, None, None] * solver.exp_ops.mH).sum(dim=-3)

            g = tuple(torch.zeros_like(p).to(y) for p in solver.options.params)

            # initialize time: time is negative-valued and sorted ascendingly during
            # backward integration.
            tstop_bwd = (-solver.tstop).flip(dims=(0,))
            saved_ini = tstop_bwd[-1] == solver.t0
            if not saved_ini:
                tstop_bwd = torch.cat((tstop_bwd, torch.zeros(1).to(tstop_bwd)))
            t0 = tstop_bwd[0].item()

            # initialize progress bar
            solver.pbar = tqdm(total=-t0, disable=not solver.options.verbose)

            # initialize the ODE routine
            (*args,) = solver.init_augmented(t0, y, a)

            # integrate the augmented equation backward between every saved state
            t = t0
            for i, tnext in enumerate(tstop_bwd[1:]):
                y, a, g, *args = solver.integrate_augmented(t, tnext, y, a, g, *args)

                if solver.options.save_states and (i < len(tstop_bwd) - 2 or saved_ini):
                    # replace y with its checkpointed version
                    y = ysave[..., -i - 2, :, :]
                    # update adjoint wrt this time point by adding dL / dy(t)
                    a += grad_y[0][..., -i - 2, :, :]

                # update adjoint wrt this time point by adding dL / de(t)
                if len(solver.exp_ops) > 0 and (i < len(tstop_bwd) - 2 or saved_ini):
                    a += (
                        grad_y[1][..., :, -i - 2, None, None] * solver.exp_ops.mH
                    ).sum(dim=-3)

                # iterate time
                t = tnext

        # close progress bar
        with warnings.catch_warnings():  # ignore tqdm precision overflow
            warnings.simplefilter('ignore', TqdmWarning)
            solver.pbar.close()

        # convert gradients of real-valued parameters to real-valued gradients
        g = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(g, solver.options.params)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, a, *g
