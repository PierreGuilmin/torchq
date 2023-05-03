from __future__ import annotations

import warnings
from abc import abstractmethod

import torch
from torch import Tensor
from tqdm.std import TqdmWarning

from ..options import Dopri45, ODEAdaptiveStep, ODEFixedStep
from ..solver import Solver
from ..utils.progress_bar import tqdm
from .adaptive_ode_integrator import DormandPrince45


class ForwardSolver(Solver):
    @abstractmethod
    def forward(self, t: float, y: Tensor) -> Tensor:
        """Iterate the quantum state forward.

        Args:
            t: Time.
            y: Quantum state, of shape `(..., m, n)`.

        Returns:
            Tensor of shape `(..., m, n)`.
        """
        pass

    def odeint(self):
        """Dispatch the ODE integration to fixed or adaptive time step subroutines."""
        if isinstance(self.options, ODEFixedStep):
            self._fixed_odeint()
        elif isinstance(self.options, ODEAdaptiveStep):
            self._adaptive_odeint()

    def _fixed_odeint(self):
        """Integrate a quantum ODE with a fixed time step ODE integrator.

        Note:
            The solver times are defined using `torch.linspace` which ensures that the
            overall solution is evolved from the user-defined time (up to an error of
            `rtol=1e-5`). However, this may induce a small mismatch between the time
            step inside `solver` and the time step inside the iteration loop. A small
            error can thus buildup throughout the ODE integration. TODO Fix this.
        """
        # get time step
        dt = self.options.dt

        # assert that `t_save` values are multiples of `dt`
        if not torch.allclose(torch.round(self.t_save / dt), self.t_save / dt):
            raise ValueError(
                'For fixed time step solvers, every value of `t_save` must be a'
                ' multiple of the time step `dt`.'
            )

        # initialize the progress bar
        tot_num_times = torch.round(self.t_save[-1] / dt).int() + 1
        pbar = tqdm(total=tot_num_times.item(), disable=not self.options.verbose)

        # run the ode routine
        t0, y = 0.0, self.y0
        for i, t1 in enumerate(self.t_save):
            # define time values
            num_times = torch.round((t1 - t0) / dt).int() + 1
            times = torch.linspace(t0, t1, num_times)

            # iterate to t1
            for t in times[:-1]:
                y = self.forward(t, y)
                pbar.update(1)

            # save
            t0 = t1
            self.save(i, y)

    def _adaptive_odeint(self):
        """Integrate a quantum ODE with an adaptive time step ODE integrator.

        This function integrates an ODE of the form `dy / dt = f(t, y)` with
        `y(0) = y0`, using a Runge-Kutta adaptive time step integrator.

        For details about the integration method, see Chapter II.4 of `Hairer et al.,
        Solving Ordinary Differential Equations I (1993), Springer Series in
        Computational Mathematics`.
        """
        # initialize the adaptive integrator
        args = (
            self.forward,
            self.options.factor,
            self.options.min_factor,
            self.options.max_factor,
            self.options.atol,
            self.options.rtol,
        )
        if isinstance(self.options, Dopri45):
            integrator = DormandPrince45(*args)

        # initialize the progress bar
        pbar = tqdm(total=self.t_save[-1].item(), disable=not self.options.verbose)

        # initialize the ODE routine
        t0 = 0.0
        f0 = integrator.f(t0, self.y0)
        dt = integrator.init_tstep(f0, self.y0, t0)
        error = 1.0
        cache = (dt, error)

        # run the ODE routine
        t, y, ft = t0, self.y0, f0
        step_counter = 0
        for i, t1 in enumerate(self.t_save):
            while t < t1:
                # update time step
                dt = integrator.update_tstep(dt, error)
                if t + dt >= t1:
                    cache = (dt, error)
                    dt = t1 - t

                # perform a single ODE integrator step of size dt
                ft_new, y_new, y_err = integrator.step(ft, y, t, dt)

                # compute estimated error of this step
                error = integrator.get_error(y_err, y, y_new)

                # update if step is accepted
                if error <= 1:
                    t, y, ft = t + dt, y_new, ft_new

                    # update the progress bar
                    with warnings.catch_warnings():  # ignore tqdm precision overflow
                        warnings.simplefilter('ignore', TqdmWarning)
                        pbar.update(dt.item())

                # raise error if max_steps reached
                step_counter += 1
                if step_counter == self.options.max_steps:
                    raise RuntimeError(
                        'Maximum number of time steps reached. Consider using lower'
                        ' order integration methods, or raising the number maximum'
                        ' number of time steps with the `options` argument.'
                    )
            # save
            dt, error = cache
            self.save(i, y)

        # close progress bar
        pbar.close()
