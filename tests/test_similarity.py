import numpy as np
import pytest
from src.similarity import cosine_loop, euclidean_loop, cosine_vectorized, euclidean_vectorized

RNG = np.random.default_rng(0)
A = RNG.random((50, 64))
B = RNG.random((50, 64))


def test_cosine_loop_vs_vectorized():
    np.testing.assert_allclose(cosine_loop(A, B), cosine_vectorized(A, B), rtol=1e-5)


def test_euclidean_loop_vs_vectorized():
    np.testing.assert_allclose(euclidean_loop(A, B), euclidean_vectorized(A, B), rtol=1e-5)


def test_cosine_identical_vectors():
    v = np.ones((5, 10))
    np.testing.assert_allclose(cosine_vectorized(v, v), np.ones(5), rtol=1e-6)


def test_euclidean_identical_vectors():
    v = np.ones((5, 10))
    np.testing.assert_allclose(euclidean_vectorized(v, v), np.zeros(5), atol=1e-7)


def test_cosine_orthogonal_vectors():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(cosine_vectorized(a, b), [0.0], atol=1e-7)


def test_cosine_antiparallel_vectors():
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    np.testing.assert_allclose(cosine_vectorized(a, b), [-1.0], atol=1e-7)


def test_euclidean_known_distance():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(euclidean_vectorized(a, b), [5.0], atol=1e-7)
