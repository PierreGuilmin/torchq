import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq


@pytest.mark.parametrize(
    ('nH', 'nT'),
    [
        ((), ()),
        ((3,), ()),
        ((3,), (1,)),
        ((3,), (3,)),
        ((3, 4), ()),
        ((3, 4), (1,)),
        ((3, 4), (3, 1)),
        ((3, 4), (4,)),
        ((3, 4), (3, 4)),
    ],
)
@pytest.mark.parametrize('H_type', ['constant', 'modulated', 'timecallable'])
def test_batching(nH, nT, H_type):
    n = 2
    tsave = jnp.linspace(0.0, 1.0, 5)
    key = jax.random.PRNGKey(84)
    key_1, key_2, key_3 = jax.random.split(key, 3)
    Ts = dq.random.real(key_3, (*nT,), min=0.5)
    if H_type == 'constant':
        H = dq.random.herm(key_1, (*nH, n, n))
    elif H_type == 'modulated':
        _H = dq.random.herm(key_1, (n, n))
        f_pref = dq.random.real(key_2, (*nH,))
        broadcast_shape = jnp.broadcast_shapes(nH, nT)
        f_pref = jnp.broadcast_to(f_pref, broadcast_shape)
        _Ts = jnp.broadcast_to(Ts, broadcast_shape)
        omega_ds = 2.0 * jnp.pi / _Ts
        H = dq.modulated(lambda t: f_pref * jnp.cos(omega_ds * t), _H)
    else:  # timecallable
        _H = dq.random.herm(key_1, (*nH, n, n))
        broadcast_shape = jnp.broadcast_shapes(nH, nT)
        _Ts = jnp.broadcast_to(Ts, broadcast_shape)
        omega_ds = 2.0 * jnp.pi / _Ts
        H = dq.timecallable(
            lambda t: jnp.einsum('...,...ij->...ij', jnp.cos(omega_ds * t), _H)
        )

    result = dq.floquet(H, Ts, tsave=tsave, safe=True)
    assert result.modes.shape == (*nH, tsave.shape[-1], n, n, 1)
    assert result.quasienergies.shape == (*nH, n)
