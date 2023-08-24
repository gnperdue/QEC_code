'''
Usage:
    python test_steane.py -v
    python test_steane.py
'''
import unittest

import sys
sys.path.append('..')   # the `general_qec` package sits above us
import numpy as np
from general_qec.qec_helpers import zero
from circuit_specific.steane_helpers import initialize_larger_steane_code
from circuit_specific.steane_helpers import phase_flip_error
from circuit_specific.steane_helpers import bit_flip_error
from circuit_specific.steane_helpers import simultaneous_steane_code


class TestSteaneCode(unittest.TestCase):

    def setUp(self):
        pass

    def test_simultaneous_steane_code(self):
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
        corrected_vector_state = simultaneous_steane_code(error_state)
        # TODO: add some good tests


if __name__ == '__main__':
    unittest.main()
