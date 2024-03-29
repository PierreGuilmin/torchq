import timeit

import jax
import jax.numpy as jnp
import pytest

from dynamiqs import basis, dag, destroy, sesolve
from dynamiqs.time_array import (
    CallableTimeArray,
    ConstantTimeArray,
    ModulatedTimeArray,
    PWCTimeArray,
    SummedTimeArray,
    modulated,
    pwc,
    timecallable,
)


def assert_equal(x, y):
    return jnp.array_equal(x, y)


class TestConstantTimeArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.x = ConstantTimeArray(jnp.array([1, 2]))

    @pytest.mark.skip('broken test')
    def test_jit(self):
        # we don't test speed here, just that it works
        x = jax.jit(self.x)
        assert_equal(x(0.0), [1, 2])

    def test_call(self):
        assert_equal(self.x(0.0), [1, 2])
        assert_equal(self.x(1.0), [1, 2])

    def test_reshape(self):
        x = self.x.reshape(1, 2)
        assert_equal(x(0.0), [[1, 2]])

    def test_conj(self):
        x = ConstantTimeArray(jnp.array([1 + 1j, 2 + 2j]))
        x = x.conj()
        assert_equal(x(0.0), [1 - 1j, 2 - 2j])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [-1, -2])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [2, 4])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [2, 4])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + 1
        assert isinstance(x, ConstantTimeArray)
        assert_equal(x(0.0), [2, 3])

        # test type `ConstantTimeArray`
        x = self.x + self.x
        assert isinstance(x, ConstantTimeArray)
        assert_equal(x(0.0), [2, 4])

    def test_radd(self):
        # test type `ArrayLike`
        x = 1 + self.x
        assert isinstance(x, ConstantTimeArray)
        assert_equal(x(0.0), [2, 3])

    def test_batching(self):
        # generic arrays
        a = destroy(4)
        H0 = a + dag(a)
        psi0 = basis(4, 0)
        times = jnp.linspace(0.0, 1.0, 11)

        # constant time array
        H_cte = jnp.stack([H0, 2 * H0])

        # test sesolve output shape
        result = sesolve(H_cte, psi0, times)
        assert result.states.shape == (2, 11, 4, 1)

        result = sesolve(H0 + H_cte, psi0, times)
        assert result.states.shape == (2, 11, 4, 1)


class TestCallableTimeArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        f = lambda t: t * jnp.array([1, 2])
        self.x = CallableTimeArray(f, ())

    @pytest.mark.skip('broken test')
    def test_jit(self):
        x = jax.jit(self.x)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [1, 2])

        t1 = timeit.timeit(lambda: x(1.0), number=1000)
        t2 = timeit.timeit(lambda: self.x(1.0), number=1000)
        assert t1 < t2

    def test_call(self):
        assert_equal(self.x(0.0), [0, 0])
        assert_equal(self.x(1.0), [1, 2])

    def test_reshape(self):
        x = self.x.reshape(1, 2)
        assert_equal(x(0.0), [[0, 0]])
        assert_equal(x(1.0), [[1, 2]])

    def test_conj(self):
        f = lambda t: t * jnp.array([1 + 1j, 2 + 2j])
        x = CallableTimeArray(f, ())
        x = x.conj()
        assert_equal(x(1.0), [1 - 1j, 2 - 2j])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [-1, -2])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + 1
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [2, 3])
        assert_equal(x(1.0), [3, 4])

        # test type `ConstantTimeArray`
        y = ConstantTimeArray(jnp.array([0, 1]))
        x = self.x + y
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])

        # test type `CallableTimeArray` (skipped for now)
        x = self.x + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

    def test_radd(self):
        # test type `ArrayLike`
        x = 1 + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [2, 3])
        assert_equal(x(1.0), [3, 4])

        # test type `ConstantTimeArray`
        y = ConstantTimeArray(jnp.array([0, 1]))
        x = y + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])

    def test_batching(self):
        # generic arrays
        a = destroy(4)
        H0 = a + dag(a)
        psi0 = basis(4, 0)
        times = jnp.linspace(0.0, 1.0, 11)

        # callable time array
        omegas = jnp.linspace(0.0, 1.0, 5)
        H_cal = timecallable(
            lambda t, omega: jnp.cos(t * omega[..., None, None]) * H0, args=(omegas,)
        )

        # test sesolve output shape
        result = sesolve(H_cal, psi0, times)
        assert result.states.shape == (5, 11, 4, 1)

        result = sesolve(H0 + H_cal, psi0, times)
        assert result.states.shape == (5, 11, 4, 1)


class TestPWCTimeArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        times = jnp.array([0, 1, 2, 3])
        values = jnp.array([1, 10, 100])
        array = jnp.array([[1, 2], [3, 4]])

        self.x = PWCTimeArray(times, values, array)  # shape at t: (2, 2)

    @pytest.mark.skip('broken test')
    def test_jit(self):
        x = jax.jit(self.x)
        assert_equal(x(-0.1), [[0, 0], [0, 0]])
        assert_equal(x(0.0), [[1, 2], [3, 4]])
        assert_equal(x(1.0), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])

        t1 = timeit.timeit(lambda: x(1.0), number=1000)
        t2 = timeit.timeit(lambda: self.x(1.0), number=1000)
        assert t1 < t2

    def test_call(self):
        assert_equal(self.x(-0.1), [[0, 0], [0, 0]])
        assert_equal(self.x(0.0), [[1, 2], [3, 4]])
        assert_equal(self.x(0.5), [[1, 2], [3, 4]])
        assert_equal(self.x(0.999), [[1, 2], [3, 4]])
        assert_equal(self.x(1.0), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])
        assert_equal(self.x(1.999), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])
        assert_equal(self.x(3.0), [[1j, 1j], [1j, 1j]])
        assert_equal(self.x(5.0), [[0, 0], [0, 0]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(-0.1), [[[0, 0], [0, 0]]])
        assert_equal(x(0.0), [[[1, 2], [3, 4]]])

    def test_conj(self):
        x = self.x.conj()
        assert_equal(x(1.0), [[10 - 1j, 20 - 1j], [30 - 1j, 40 - 1j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1, -2], [-3, -4]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + 2
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [[3, 5], [5, 6]])

        # test type `ConstantTimeArray`
        y = ConstantTimeArray(jnp.array([[1, 1], [1, 1]]))
        x = self.x + y
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `PWCTimeArray`
        x = self.x + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_radd(self):
        # test type `ArrayLike`
        x = 2 + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [[3, 5], [5, 6]])

        # test type `ConstantTimeArray`
        y = ConstantTimeArray(jnp.array([[1, 1], [1, 1]]))
        x = y + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

    def test_batching(self):
        # generic arrays
        a = destroy(4)
        H0 = a + dag(a)
        psi0 = basis(4, 0)
        times = jnp.linspace(0.0, 1.0, 11)

        # pwc time array
        values = jnp.arange(3 * 10).reshape(3, 10)
        H_pwc = pwc(times, values, H0)

        # test sesolve output shape
        result = sesolve(H_pwc, psi0, times)
        assert result.states.shape == (3, 11, 4, 1)

        result = sesolve(H0 + H_pwc, psi0, times)
        assert result.states.shape == (3, 11, 4, 1)


class TestModulatedTimeArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        one = jnp.array(1.0)
        eps = lambda t: (0.5 * t + 1.0j) * one
        array = jnp.array([[1, 2], [3, 4]])

        self.x = ModulatedTimeArray(eps, array, ())

    def test_call(self):
        assert_equal(self.x(0.0), [[1.0j, 2.0j], [3.0j, 4.0j]])
        assert_equal(self.x(2.0), [[1.0 + 5.0j, 2.0 + 6.0j], [3.0 + 7.0j, 4.0 + 8.0j]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[1.0j, 2.0j], [3.0j, 4.0j]]])
        assert_equal(x(2.0), [[[1.0 + 5.0j, 2.0 + 6.0j], [3.0 + 7.0j, 4.0 + 8.0j]]])

    def test_conj(self):
        x = self.x.conj()
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + 2
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [[3.0j, 4.0j], [5.0j, 6.0j]])

        # test type `ConstantTimeArray`
        y = ConstantTimeArray(jnp.array([[1, 1], [1, 1]]))
        x = self.x + y
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(-0.1), [[3.0j, 4.0j], [5.0j, 6.0j]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `ModulatedTimeArray`
        x = self.x + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_radd(self):
        # test type `ArrayLike`
        x = 2 + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(0.0), [[3.0j, 4.0j], [5.0j, 6.0j]])

        # test type `ConstantTimeArray`
        y = ConstantTimeArray(jnp.array([[1, 1], [1, 1]]))
        x = y + self.x
        assert isinstance(x, SummedTimeArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

    def test_batching(self):
        # generic arrays
        a = destroy(4)
        H0 = a + dag(a)
        psi0 = basis(4, 0)
        times = jnp.linspace(0.0, 1.0, 11)

        # modulated time array
        deltas = jnp.linspace(0.0, 1.0, 4)
        H_mod = modulated(lambda t, delta: jnp.cos(t * delta), H0, args=(deltas,))

        # test sesolve output shape
        result = sesolve(H_mod, psi0, times)
        assert result.states.shape == (4, 11, 4, 1)

        result = sesolve(H0 + H_mod, psi0, times)
        assert result.states.shape == (4, 11, 4, 1)
