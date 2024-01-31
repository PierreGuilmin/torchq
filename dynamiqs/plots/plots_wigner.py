from __future__ import annotations

import pathlib
import shutil

import imageio
import IPython.display as ipy
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from tqdm import tqdm

from ..utils.wigners import wigner
from .utils import add_colorbar, colors, figax, gridplot, optional_ax

__all__ = ['plot_wigner', 'plot_wigner_mosaic', 'plot_wigner_gif']


@optional_ax
def plot_wigner_data(
    wigner: ArrayLike,
    xmax: float,
    ymax: float,
    *,
    ax: Axes | None = None,
    vmax: float = 2 / jnp.pi,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    w = jnp.asarray(wigner)

    # set plot norm
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # clip to avoid rounding errors
    w = w.clip(vmin, vmax)

    # plot
    ax.imshow(
        w,
        cmap=cmap,
        norm=norm,
        origin='lower',
        aspect='equal',
        interpolation=interpolation,
        extent=[-xmax, xmax, -ymax, ymax],
    )

    # axis label
    ax.set(xlabel=r'$\mathrm{Re}(\alpha)$', ylabel=r'$\mathrm{Im}(\alpha)$')

    if colorbar and not clear:
        cax = add_colorbar(ax, cmap, norm)
        if vmax == 2 / jnp.pi:
            cax.set_yticks([vmin, 0.0, vmax], labels=[r'$-2/\pi$', r'$0$', r'$2/\pi$'])

    if cross:
        ax.axhline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)
        ax.axvline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)

    if clear:
        ax.grid(False)
        ax.axis(False)


@optional_ax
def plot_wigner(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    r"""Plot the Wigner function of a state.

    Warning:
        Documentation redaction in progress.

    Note:
        Choose a diverging colormap `cmap` for better results.

    Warning:
        The axis scaling is chosen so that a coherent state $\ket{\alpha}$ lies at the
        coordinates $(x,y)=(\mathrm{Re}(\alpha),\mathrm{Im}(\alpha))$, which is
        different from the default behaviour of `qutip.plot_wigner()`.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot_wigner(psi)
        >>> renderfig('plot_wigner_coh')

        ![plot_wigner_coh](/figs-code/plot_wigner_coh.png){.fig-half}

        >>> psi = dq.unit(dq.coherent(16, 2) + dq.coherent(16, -2))
        >>> dq.plot_wigner(psi, xmax=4.0, ymax=2.0, colorbar=False)
        >>> renderfig('plot_wigner_cat')

        ![plot_wigner_cat](/figs-code/plot_wigner_cat.png){.fig-half}

        >>> psi = dq.unit(dq.fock(2, 0) + dq.fock(2, 1))
        >>> dq.plot_wigner(psi, xmax=2.0, cross=True)
        >>> renderfig('plot_wigner_01')

        ![plot_wigner_01](/figs-code/plot_wigner_01.png){.fig-half}

        >>> psi = dq.unit(sum(dq.coherent(32, 3 * a) for a in [1, 1j, -1, -1j]))
        >>> dq.plot_wigner(psi, npixels=201, clear=True)
        >>> renderfig('plot_wigner_4legged')

        ![plot_wigner_4legged](/figs-code/plot_wigner_4legged.png){.fig-half}
    """
    state = jnp.asarray(state)

    ymax = xmax if ymax is None else ymax
    _, _, w = wigner(state, xmax, ymax, npixels)

    plot_wigner_data(
        w,
        xmax,
        ymax,
        ax=ax,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=colorbar,
        cross=cross,
        clear=clear,
    )


def plot_wigner_mosaic(
    states: ArrayLike,
    *,
    n: int = 8,
    nrows: int = 1,
    w: float = 3.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
):
    r"""Plot the Wigner function of multiple states in a mosaic arrangement.

    Warning:
        Documentation redaction in progress.

    See [`dq.plot_wigner()`][dynamiqs.plot_wigner] for more details.

    Examples:
        >>> psis = [dq.fock(3, i) for i in range(3)]
        >>> dq.plot_wigner_mosaic(psis)
        >>> renderfig('plot_wigner_mosaic_fock')

        ![plot_wigner_mosaic_fock](/figs-code/plot_wigner_mosaic_fock.png){.fig}

        >>> # n = 16
        >>> # a = dq.destroy(n)
        >>> # H = dq.zero(n)
        >>> # jump_ops = [a @ a - 4.0 * dq.eye(n)]
        >>> # psi0 = dq.coherent(n, 0)
        >>> # tsave = np.linspace(0, 1.0, 101)
        >>> # result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> # dq.plot_wigner_mosaic(result.states, n=6, xmax=4.0, ymax=2.0)
        >>> # renderfig('plot_wigner_mosaic_cat')

        <!-- ![plot_wigner_mosaic_cat](/figs-code/plot_wigner_mosaic_cat.png){.fig} -->

        >>> # n = 16
        >>> # a = dq.destroy(n)
        >>> # H = dq.dag(a) @ dq.dag(a) @ a @ a  # Kerr Hamiltonian
        >>> # psi0 = dq.coherent(n, 2)
        >>> # tsave = np.linspace(0, np.pi, 101)
        >>> # result = dq.sesolve(H, psi0, tsave)
        >>> # dq.plot_wigner_mosaic(result.states, n=25, nrows=5, xmax=4.0)
        >>> # renderfig('plot_wigner_mosaic_kerr')

        <!-- ![plot_wigner_mosaic_kerr](/figs-code/plot_wigner_mosaic_kerr.png){.fig} -->
    """  # noqa: E501
    # todo: fix examples
    states = jnp.asarray(states)

    nstates = len(states)
    if nstates < n:
        n = nstates

    # create grid of plot
    _, axs = gridplot(
        n,
        nrows=nrows,
        w=w,
        h=h,
        gridspec_kw=dict(wspace=0, hspace=0),
        sharex=True,
        sharey=True,
    )

    ymax = xmax if ymax is None else ymax
    selected_indexes = np.linspace(0, nstates, n, dtype=int)
    _, _, w = wigner(states[selected_indexes], xmax, ymax, npixels)

    # plot individual wigner
    for i, ax in enumerate(axs):
        plot_wigner_data(
            w[i],
            ax=ax,
            xmax=xmax,
            ymax=ymax,
            vmax=vmax,
            cmap=cmap,
            interpolation=interpolation,
            colorbar=False,
            cross=cross,
            clear=False,
        )
        ax.set(xlabel='', ylabel='', xticks=[], yticks=[])


def plot_wigner_gif(
    states: ArrayLike,
    gif_duration: float = 5.0,
    fps: int = 10,
    w: float = 5.0,
    h: float = 5.0,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
    clear: bool = False,
):
    """Plot a gif of the Wigner function of multiple states.

    The function saves a gif of the Wigner function plot and displays it.

    Parameters:
        states (ArrayLike): The quantum states to be plotted.
        gif_duration (float): The length of the gif in seconds.
        fps (int): The frames per second of the gif.
        w (float): The width of the plot.
        h (float): The height of the plot.
        xmax (float): The maximum x value of the plot.
        ymax (float | None): The maximum y value of the plot. If None, it is set to
            xmax.
        vmax (float): The maximum value of the colorbar.
        npixels (int): The number of pixels in the plot.
        cmap (str): The colormap to be used.
        interpolation (str): The interpolation method to be used.
        cross (bool): If True, a cross is plotted at the origin.
        clear (bool): If True, the axes are cleared.
    """
    if ymax is None:
        ymax = xmax

    states = jnp.asarray(states)

    nframes = int(gif_duration * fps)
    tmpdir = './.tmp/dynamiqs'
    tmpdir = pathlib.Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    selected_indexes = np.linspace(0, len(states), nframes, dtype=int)
    _, _, wigners = wigner(states[selected_indexes], xmax, ymax, npixels)

    tmpdir.mkdir(exist_ok=True)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        fig, ax = figax(w=w, h=h)

        plot_wigner_data(
            wigners[frame_idx],
            ax=ax,
            xmax=xmax,
            ymax=ymax,
            vmax=vmax,
            cmap=cmap,
            interpolation=interpolation,
            colorbar=False,
            cross=cross,
            clear=clear,
        )

        filename = tmpdir / f'tmp-{frame_idx}.png'
        fig.savefig(filename)
        plt.close()
        frame = imageio.v2.imread(filename)
        frames.append(frame)

    if tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)

    filename = 'wigner.gif'
    imageio.mimwrite(filename, frames, format='GIF', duration=1 / fps)
    ipy.display(ipy.Image(filename))
