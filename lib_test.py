from typing import Tuple

import unittest
from parameterized import parameterized 

import lib
import jax.numpy as jnp
from numpy import testing

class Test(unittest.TestCase):

    @parameterized.expand([
        [1, 0, 10, lib.Operation.ADDITION, 2, 1, (
            jnp.array([[[6], [2]]]), jnp.array([[8]])
        )],
        [1, 0, 10, lib.Operation.MULTIPLICATION, 2, 1, (
            jnp.array([[[6], [2]]]), jnp.array([[12]])
        )],
    ])
    def test_make_synthetic_examples(
        self,
        num_examples: int,
        minval: int,
        maxval: int,
        operation: lib.Operation | str,
        num_operations: int,
        batch_size: int,
        expected: Tuple[jnp.ndarray, jnp.ndarray],
        seed: int = 0
    ):
        actual = lib.make_synthetic_examples(
            num_examples=num_examples,
            minval=minval,
            maxval=maxval,
            operation=operation,
            num_operations=num_operations,
            batch_size=batch_size,
            seed=seed,
        )
        testing.assert_array_equal(actual[0], expected[0])
        testing.assert_array_equal(actual[1], expected[1])

if __name__ == "__main__":
    unittest.main()