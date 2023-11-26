from __future__ import annotations

import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

from ..utils.tensor_types import ArrayLike, to_numpy
from .utils import add_colorbar, bra_ticks, integer_ticks, ket_ticks, optax

__all__ = ['plot_hinton']


def _plot_squares(
    ax: Axes,
    areas: np.ndarray,
    colors: np.ndarray,
    offsets: np.ndarray,
    ecolor: str = 'white',
    ewidth: float = 0.5,
):
    # areas: 1D array (n) with real values in [0, 1]
    # colors: 2D array (n, 4) with RGBA values
    # offsets: 2D array (n, 2) with real values in R

    # compute squares side length
    sides = np.sqrt(areas)

    # for efficiency we only keep squares with non-negligible side length >= 0.01 (side
    # length is in [0, 1])
    mask = sides >= 0.01
    sides, colors, offsets = sides[mask], colors[mask], offsets[mask]

    # compute squares corner coordinates
    corners = offsets - sides[..., None] / 2

    patch_list = [
        patches.Rectangle(xy, side, side, facecolor=color)
        for xy, side, color in zip(corners, sides, colors)
    ]

    squares = PatchCollection(
        patch_list, match_original=True, edgecolor=ecolor, linewidth=ewidth
    )

    ax.add_collection(squares)


@optax
def _plot_hinton(
    areas: np.ndarray,
    colors: np.ndarray,
    colors_vmin: float,
    colors_vmax: float,
    cmap: str,
    *,
    ax: Axes | None = None,
    colorbar: bool = True,
    allticks: bool = True,
    ecolor: str = 'white',
    ewidth: float = 0.5,
):
    # areas: 2D array (n, n) with real values in [0, 1]
    # colors: 2D array (n, n) with real values in [0, 1]

    areas = areas.clip(0.0, 1.0)
    colors = colors.clip(0.0, 1.0)

    # === set axes
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='both', direction='out')
    n = areas.shape[0]
    ax.set(xlim=(-0.5, n - 1 + 0.5), ylim=(-0.5, n - 1 + 0.5))
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    integer_ticks(ax.xaxis, n, all=allticks)
    integer_ticks(ax.yaxis, n, all=allticks)

    # === plot squares
    # squares coordinates
    offsets = np.array(list(np.ndindex(areas.shape)))
    # squares areas
    areas = areas.T.flatten()
    # squares colors
    cmap = mpl.colormaps.get_cmap(cmap)
    colors = cmap(colors.T).reshape(-1, 4)
    _plot_squares(ax, areas, colors, offsets, ecolor=ecolor, ewidth=ewidth)

    # === colorbar
    if colorbar:
        norm = Normalize(colors_vmin, colors_vmax)
        cax = add_colorbar(ax, cmap, norm, size='4%', pad='4%')
        if colors_vmin == -np.pi and colors_vmax == np.pi:
            cax.set_yticks([-np.pi, 0.0, np.pi], labels=[r'$-\pi$', r'$0$', r'$\pi$'])


@optax
def plot_hinton(
    x: ArrayLike,
    *,
    ax: Axes | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: 'bool' = True,
    allticks: bool = False,
    tickslabel: list[str] | None = None,
    ecolor: str = 'white',
    ewidth: float = 0.5,
    clear: bool = False,
):
    """Plot a Hinton diagram.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> rho = dq.coherent_dm(16, 2.0)
        >>> dq.plot_hinton(rho.abs())
        >>> renderfig('plot_hinton_coherent')

        ![plot_hinton_coherent](/figs-code/plot_hinton_coherent.png){.fig-half}

        >>> a = dq.destroy(16)
        >>> H = a.mH @ a + 2.0 * (a + a.mH)
        >>> dq.plot_hinton(H.abs())
        >>> renderfig('plot_hinton_hamiltonian')

        ![plot_hinton_hamiltonian](/figs-code/plot_hinton_hamiltonian.png){.fig-half}

        >>> cnot = torch.tensor(
        ...     [[1, 0, 0, 0],
        ...      [0, 1, 0, 0],
        ...      [0, 0, 0, 1],
        ...      [0, 0, 1, 0]],
        ...  )
        >>> dq.plot_hinton(cnot, tickslabel=['00', '01', '10', '11'])
        >>> renderfig('plot_hinton_cnot')

        ![plot_hinton_cnot](/figs-code/plot_hinton_cnot.png){.fig-half}

        >>> x = dq.rand_complex((16, 16))
        >>> dq.plot_hinton(x)
        >>> renderfig('plot_hinton_rand_complex')

        ![plot_hinton_rand_complex](/figs-code/plot_hinton_rand_complex.png){.fig-half}

        >>> _, axs = dq.gridplot(2)
        >>> psi = dq.unit(dq.fock(4, 0) - dq.fock(4, 2))
        >>> dq.plot_hinton(dq.todm(psi), ax=next(axs))
        >>> rho = dq.unit(dq.fock_dm(4, 0) + dq.fock_dm(4, 2))
        >>> dq.plot_hinton(rho, ax=next(axs))
        >>> renderfig('plot_hinton_fock02')

        ![plot_hinton_fock02](/figs-code/plot_hinton_fock02.png){.fig-half}

        >>> _, axs = dq.gridplot(2)
        >>> x = np.random.uniform(-1.0, 1.0, (10, 10))
        >>> dq.plot_hinton(x, ax=next(axs), vmin=-1.0, vmax=1.0)
        >>> dq.plot_hinton(np.abs(x), ax=next(axs), cmap='Greys', vmax=1.0, ecolor='black')
        >>> renderfig('plot_hinton_real')

        ![plot_hinton_real](/figs-code/plot_hinton_real.png){.fig-half}

        >>> x = np.random.uniform(-1.0, 1.0, (100, 100))
        >>> dq.plot_hinton(x, vmin=-1.0, vmax=1.0, ewidth=0, clear=True, w=20)
        >>> renderfig('plot_hinton_large')

        ![plot_hinton_large](/figs-code/plot_hinton_large.png){.fig}
    """  # noqa: E501

    x = to_numpy(x)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(
            f'Argument `x` must be a 2D square array, but has shape {x.shape}.'
        )

    # set different defaults, areas and colors for real matrix, positive real matrix
    # and complex matrix
    if np.isrealobj(x):
        # x: 2D array with real data in [vmin, vmax]

        all_positive = np.all(x >= 0)
        if cmap is None:
            # sequential colormap for positive data, diverging colormap otherwise
            cmap = 'Blues' if all_positive else 'dq'
        if vmin is None:
            vmin = 0.0 if all_positive else np.min(x)

        vmax = np.max(x) if vmax is None else vmax

        # areas: absolute value of x
        area_max = max(abs(vmin), abs(vmax))
        areas = Normalize(0.0, area_max)(np.abs(x))

        # colors: value of x
        colors = Normalize(vmin, vmax)(x)
        colors_vmin, colors_vmax = vmin, vmax
    elif np.iscomplexobj(x):
        # x: 2D array with complex data

        # cyclic colormap for the phase
        cmap = 'dq_cyclic' if cmap is None else cmap

        # areas: magnitude of x
        magnitude = np.abs(x)
        areas_max = np.max(magnitude) if vmax is None else vmax
        areas = Normalize(0.0, areas_max)(magnitude)

        # colors: phase of x
        phase = np.angle(x)
        colors = Normalize(-np.pi, np.pi)(phase)
        colors_vmin, colors_vmax = -np.pi, np.pi

    if clear:
        colorbar = False

    if tickslabel is not None:
        allticks = True

    _plot_hinton(
        areas,
        colors,
        colors_vmin,
        colors_vmax,
        cmap,
        ax=ax,
        colorbar=colorbar,
        allticks=allticks,
        ecolor=ecolor,
        ewidth=ewidth,
    )

    # set ticks label format
    if tickslabel is not None:
        ax.xaxis.set_ticklabels(tickslabel)
        ax.yaxis.set_ticklabels(tickslabel)

    ket_ticks(ax.xaxis)
    bra_ticks(ax.yaxis)

    if clear:
        ax.axis(False)
