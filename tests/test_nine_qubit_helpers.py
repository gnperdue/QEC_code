'''
Usage:
    python test_nine_qubit_helpers.py
'''
import unittest
import random
import logging
import sys
import numpy as np
from general_qec.qec_helpers import one, zero, superpos
from general_qec.qec_helpers import bit_flip_error, phase_flip_error
from general_qec.qec_helpers import print_state_info
# from general_qec.qec_helpers import ancilla_reset
from circuit_specific.nine_qubit_helpers import nine_qubit_initialize_logical_state
from circuit_specific.nine_qubit_helpers import nine_qubit_bit_correction
from circuit_specific.nine_qubit_helpers import nine_qubit_phase_correction

LOGGER = logging.getLogger(__name__)

THREE_ZERO = np.kron(zero, np.kron(zero, zero))
THREE_ONE = np.kron(one, np.kron(one, one))
THREE_PLUS = THREE_ZERO + THREE_ONE
THREE_MINUS = THREE_ZERO - THREE_ONE
LOGICAL_ZERO = 1. / np.sqrt(8.) * np.kron(
    THREE_PLUS, np.kron(THREE_PLUS, THREE_PLUS)
)
LOGICAL_ONE = 1. / np.sqrt(8.) * np.kron(
    THREE_MINUS, np.kron(THREE_MINUS, THREE_MINUS)
)
LOGICAL_SUPERPOS = 1. / np.sqrt(2.) * (LOGICAL_ZERO + LOGICAL_ONE)

ANCILLAE = np.kron(zero, zero)
FULL_ZERO = np.kron(LOGICAL_ZERO, ANCILLAE)
FULL_ONE = np.kron(LOGICAL_ONE, ANCILLAE)
FULL_SUPERPOS = np.kron(LOGICAL_SUPERPOS, ANCILLAE)


class TestNineQubitHelpers(unittest.TestCase):
    """Tests for nine-qubit code helper functions."""

    def test_nine_qubit_initialize_logical_state(self):
        """
        Test nine-qubit code initialization.
        """
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # -
        initialized_zero_state = nine_qubit_initialize_logical_state(zero)
        self.assertTrue(np.allclose(initialized_zero_state, FULL_ZERO))
        initialized_one_state = nine_qubit_initialize_logical_state(one)
        self.assertTrue(np.allclose(initialized_one_state, FULL_ONE))
        initialized_superpos_state = nine_qubit_initialize_logical_state(superpos)
        self.assertTrue(np.allclose(initialized_superpos_state, FULL_SUPERPOS))

    def test_nine_qubit_bit_correction(self):
        """
        Test nine-qubit code bit correction.
        """
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # -
        random.seed(11)
        initialized_superpos_state = nine_qubit_initialize_logical_state(superpos)
        print("init state")
        print_state_info(initialized_superpos_state, 11)
        bit_error_state, bit_index = bit_flip_error(initialized_superpos_state, 9)
        print("bit error state")
        print_state_info(bit_error_state, 11)
        print(bit_index)
        self.assertEqual(bit_index, 6)
        corrected_state = nine_qubit_bit_correction(bit_error_state)
        print("corrected state")
        print_state_info(corrected_state, 11)
        self.assertTrue(np.allclose(initialized_superpos_state, corrected_state))

    def test_nine_qubit_phase_correction(self):
        """
        Test nine-qubit code phase correction.
        """
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # -
        random.seed(10)
        initialized_superpos_state = nine_qubit_initialize_logical_state(superpos)
        # print("init state")
        # print_state_info(initialized_superpos_state, 11)
        phase_error_state, phase_index = phase_flip_error(initialized_superpos_state, 9)
        # print("phase error state")
        # print_state_info(phase_error_state, 11)
        # print(phase_index)
        self.assertEqual(phase_index, 8)
        corrected_state = nine_qubit_phase_correction(phase_error_state)
        # print("corrected state")
        # print_state_info(corrected_state, 11)
        self.assertTrue(np.allclose(initialized_superpos_state, corrected_state))


if __name__ == '__main__':
    unittest.main()
