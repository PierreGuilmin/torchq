from __future__ import annotations

from math import pi, sqrt

import torch

import dynamiqs as dq


class CatCNOT(OpenSystem):
    def __init__(
        self,
        N: int = 32,
        num_tslots: int = 100,
        alpha: float = 2.0,
        kappa2: float = 1.0 * MHz,
        T: float = 200 * ns,
    ):
        self.N = N
        self.num_tslots = num_tslots
        self.kappa2 = kappa2
        self.T = T

        # cnot drive amplitude
        self.g = pi / (4 * alpha * T)

        # Hamiltonian
        ac = dq.tensprod(dq.destroy(N), dq.eye(N))
        at = dq.tensprod(dq.eye(N), dq.destroy(N))
        i = dq.tensprod(dq.eye(N), dq.eye(N))
        self.H = self.g * (ac + ac.mH) @ (at.mH @ at - alpha**2 * i)

        # jump operator
        self.jump_ops = [sqrt(kappa2) * (ac @ ac - alpha**2 * i)]

        # initial state
        plus = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
        self.y0 = dq.tensprod(plus, plus)

        # tsave
        self.tsave = torch.linspace(0, self.T, self.num_tslots + 1)

    def to(self, dtype: torch.dtype, device: torch.device):
        super().to(dtype=dtype, device=device)
        self.H = self.H.to(dtype=dtype, device=device)
