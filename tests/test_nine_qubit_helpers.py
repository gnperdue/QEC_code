'''
Usage:
    python test_nine_qubit_helpers.py
'''
import unittest
import random
import logging
import sys
import numpy as np
from general_qec.qec_helpers import one, zero
# from general_qec.qec_helpers import ancilla_reset
from circuit_specific.nine_qubit_helpers import nine_qubit_initialize_logical_state

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

ANCILLAE = np.kron(zero, zero)
FULL_ZERO = np.kron(LOGICAL_ZERO, ANCILLAE)
FULL_ONE = np.kron(LOGICAL_ONE, ANCILLAE)


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


if __name__ == '__main__':
    unittest.main()
