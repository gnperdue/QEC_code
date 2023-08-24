'''
Usage:
    python test_steane.py -v
    python test_steane.py
'''
import unittest
import random

import sys
sys.path.append('..')   # the `general_qec` package sits above us
import numpy as np
from general_qec.qec_helpers import zero
from general_qec.qec_helpers import vector_state_to_bit_state
from general_qec.gates import sigma_I, sigma_x, sigma_y, sigma_z
from circuit_specific.steane_helpers import initialize_larger_steane_code
from circuit_specific.steane_helpers import phase_flip_error
from circuit_specific.steane_helpers import bit_flip_error
from circuit_specific.steane_helpers import simultaneous_steane_code


class TestFiveQubitStabilizer(unittest.TestCase):

    def setUp(self) -> None:
        self.zero_state = \
            np.kron(zero, 
                    np.kron(zero, 
                            np.kron(zero, 
                                    np.kron(zero, zero))))

        # Set the 4 stabilizer operators
        self.k_one = np.kron(
            sigma_x, np.kron(
            sigma_z, np.kron(
            sigma_z, np.kron(
            sigma_x, sigma_I))))
        self.k_two = np.kron(
            sigma_I, np.kron(
            sigma_x, np.kron(
            sigma_z, np.kron(
            sigma_z, sigma_x))))
        self.k_three = np.kron(
            sigma_x, np.kron(
            sigma_I, np.kron(
            sigma_x, np.kron(
            sigma_z, sigma_z))))
        self.k_four = np.kron(
            sigma_z, np.kron(
            sigma_x, np.kron(
            sigma_I, np.kron(
            sigma_x, sigma_z))))

        # Set the logical Z operator to fix the logical state
        self.z_bar = np.kron(
            sigma_z, np.kron(
            sigma_z, np.kron(
            sigma_z, np.kron(
            sigma_z, sigma_z))))
        
        # Create and apply the stebilizer operation on the 5 qubit system
        self.operation = np.dot(
            (np.identity(2**5) + self.k_one), np.dot(
            (np.identity(2**5) + self.k_two), np.dot(
            (np.identity(2**5) + self.k_three), (np.identity(2**5) + self.k_four))))
        self.initialized_state = 0.25* np.dot(self.operation, self.zero_state)

        return super().setUp()

    def test_vector_state_to_bit_state(self):
        bits, indexes, state = vector_state_to_bit_state(self.initialized_state, 5)
        self.assertAlmostEqual(np.sum(state**2), 1.0)
        self.assertEqual(bits.shape, (16,))
        self.assertEqual(indexes.shape, (16,))


class TestSteaneCode(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_simultaneous_steane_code(self):
        random.seed(10)
        n_qubits = 7  # number of data qubits in our system
        n_ancilla = 6 # for "simultaneous" Steane
        n_qtotal = n_qubits + n_ancilla
        initial_state = np.kron(
              zero, np.kron(
              zero, np.kron(
              zero, np.kron(
              zero, np.kron(
              zero, np.kron(zero, zero))))))
        self.assertEqual(initial_state.shape, (2**n_qubits,))
        final_vector_state = initialize_larger_steane_code(
            np.sqrt(8) * initial_state)
        self.assertEqual(final_vector_state.shape, (2**n_qtotal,))
        error_state = phase_flip_error(
            bit_flip_error(final_vector_state, n_qtotal)[0], n_qtotal)[0]
        self.assertEqual(error_state.shape, (2**n_qtotal,))
#         corrected_vector_state = simultaneous_steane_code(error_state)
        # TODO: add some good tests


if __name__ == '__main__':
    unittest.main()
