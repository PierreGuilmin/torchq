import jax
import jax.numpy as jnp

from dynamiqs.sparse import *


class TestSparseDIA:
    def test_matmul(self, rtol=1e-05, atol=1e-08):
        N = 4
        keyA, keyB = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
        matrixA = jax.random.normal(keyA, (N, N))
        matrixB = jax.random.normal(keyB, (N, N))

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = (sparseA @ sparseB).to_dense()
        out_dia_dense = sparseA @ matrixB
        out_dense_dia = matrixA @ sparseB
        out_dense_dense = matrixA @ matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_add(self, rtol=1e-05, atol=1e-08):
        N = 4
        keyA, keyB = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
        matrixA = jax.random.normal(keyA, (N, N))
        matrixB = jax.random.normal(keyB, (N, N))

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = (sparseA + sparseB).to_dense()
        out_dia_dense = sparseA + matrixB
        out_dense_dia = matrixA + sparseB
        out_dense_dense = matrixA + matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_transform(self, rtol=1e-05, atol=1e-08):
        N = 4
        key = jax.random.PRNGKey(1)
        matrix = jax.random.normal(key, (N, N))
        assert jnp.allclose(matrix, to_sparse(matrix).to_dense(), rtol=rtol, atol=atol)
