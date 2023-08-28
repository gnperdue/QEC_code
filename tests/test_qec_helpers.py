'''
Usage:
    python test_qec_helpers.py
'''
import unittest
import random

import logging
LOGGER = logging.getLogger(__name__)

import sys
sys.path.append('..')   # the `general_qec` package sits above us

import numpy as np
from general_qec.qec_helpers import one, zero, superpos
from general_qec.gates import sigma_y
from general_qec.qec_helpers import ancilla_reset
from general_qec.qec_helpers import collapse_ancilla
from general_qec.qec_helpers import collapse_dm
from general_qec.qec_helpers import remove_small_values
from general_qec.qec_helpers import vector_state_to_bit_state
from general_qec.qec_helpers import CNOT_gate_tot


class TestHelpers(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    def test_vector_state_to_bit_state(self):
        test_state1 = np.kron(one, zero)
        log_bit, index_of_element, logical_state = vector_state_to_bit_state(test_state1, 2)
        self.assertEqual(log_bit.shape, (1,))
        self.assertEqual(index_of_element.shape, (1,))
        self.assertEqual(log_bit[0], '10')
        self.assertEqual(index_of_element[0], 2.0)
        self.assertAlmostEqual(logical_state[2], 1.0)
        test_state2 = np.kron(superpos, superpos)
        log_bit, index_of_element, logical_state = vector_state_to_bit_state(test_state2, 2)
        self.assertEqual(log_bit.shape, (4,))
        self.assertEqual(index_of_element.shape, (4,))
        self.assertEqual(log_bit[3], '11')
        self.assertEqual(index_of_element[2], 2.0)
        self.assertAlmostEqual(logical_state[1], 0.5)

    def test_ancilla_functions(self):
        test_state1 = np.kron(np.kron(superpos, superpos), superpos)
        random.seed(10)  # fix the collapsed state
        collapsed_vector_state = collapse_ancilla(test_state1, 1)
        self.assertEqual(collapsed_vector_state.shape, (8,))
        self.assertAlmostEqual(collapsed_vector_state[1], 0.5+0.0j)
        reset_state = ancilla_reset(collapsed_vector_state, 1)
        self.assertEqual(reset_state.shape, (8,))
        self.assertAlmostEqual(reset_state[0], 0.5+0.0j)
        three_qubit = np.kron(np.kron(zero, zero), zero)
        test_state2 = np.kron(np.kron(three_qubit, superpos), superpos)
        random.seed(10)  # fix the collapsed state
        collapsed_vector_state = collapse_ancilla(test_state2, 2)
        self.assertEqual(collapsed_vector_state.shape, (32,))
        self.assertAlmostEqual(collapsed_vector_state[2], 1.0+0.0j)
        reset_state = ancilla_reset(collapsed_vector_state, 2)
        self.assertEqual(reset_state.shape, (32,))
        self.assertAlmostEqual(reset_state[0], 1.0+0.0j)

    def test_remove_small_values(self):
        x = np.array([1, 1e-16, 1e-16, 1e-16])
        y = np.array([1, 0, 0, 0])
        self.assertTrue(np.all(remove_small_values(x) == y))
        self.assertTrue(np.all(remove_small_values(x, tolerance=1e-17) == x))

    def test_CNOT_gate_tot(self):
        self.assertEqual(CNOT_gate_tot(5, 3), 6)
        self.assertEqual(CNOT_gate_tot(3, 7), 14)

    def test_collapse_dm(self):
        random.seed(13)  # fix the collapsed state
        initial_state = np.kron(superpos, superpos)
        initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
        collapsed_state = collapse_dm(initial_rho)
        self.assertTrue(np.all(collapsed_state == np.array([0, 1, 0, 0])))
        random.seed(13)  # fix the collapsed state
        initial_state = np.kron(np.kron(zero, superpos), one)
        initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
        collapsed_state = collapse_dm(initial_rho)
        self.assertEqual(collapsed_state[1], 1)
        self.assertEqual(np.sum(collapsed_state), 1)
        random.seed(13)  # fix the collapsed state
        initial_state = np.kron(np.matmul(sigma_y, zero), superpos)
        initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
        collapsed_state = collapse_dm(initial_rho)
        self.assertEqual(collapsed_state[2], 1)
        self.assertEqual(np.sum(collapsed_state), 1)


if __name__ == '__main__':
    unittest.main()