'''
Usage:
    python test_steane_helpers.py
'''
import unittest
import random
import logging
import sys
import numpy as np
from general_qec.qec_helpers import one, zero, superpos
from general_qec.qec_helpers import ancilla_reset
from general_qec.qec_helpers import print_state_info
from general_qec.qec_helpers import vector_state_to_bit_state
from general_qec.gates import sigma_I, sigma_x, sigma_y, sigma_z
# from circuit_specific.steane_helpers import \  # TODO use these later...
#     k_one, k_two, k_three, \
#     k_four, k_five, k_six
from circuit_specific.steane_helpers import bit_flip_error
from circuit_specific.steane_helpers import initialize_larger_steane_code
from circuit_specific.steane_helpers import initialize_steane_logical_state
from circuit_specific.steane_helpers import phase_flip_error
from circuit_specific.steane_helpers import simultaneous_steane_code
from circuit_specific.steane_helpers import steane_phase_correction
from circuit_specific.steane_helpers import steane_bit_correction

LOGGER = logging.getLogger(__name__)

ZERO_STATE7Q = \
    np.kron(zero,
            np.kron(zero,
                    np.kron(zero,
                            np.kron(zero,
                                    np.kron(zero, np.kron(zero, zero)))))
    )
ONE_STATE7Q = \
    np.kron(one,
            np.kron(one,
                    np.kron(one,
                            np.kron(one,
                                    np.kron(one, np.kron(one, one)))))
    )
SUPERPOS_STATE7Q = \
    np.kron(superpos,
            np.kron(superpos,
                    np.kron(superpos,
                            np.kron(superpos,
                                    np.kron(superpos, np.kron(superpos, superpos)))))
    )


class TestFiveQubitStabilizer(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the five qubit stabilizer functions."""

    def setUp(self) -> None:
        self.zero_state = np.kron(
            zero, np.kron(
            zero, np.kron(
            zero, np.kron(
            zero, zero))))

        # Set the 4 stabilizer operators for the 5 qubit code
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
        """Tests for `vector_state_to_bit_state()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        bits, indexes, state = vector_state_to_bit_state(self.initialized_state, 5)
        self.assertAlmostEqual(np.sum(state**2), 1.0)
        self.assertEqual(bits.shape, (16,))
        self.assertEqual(indexes.shape, (16,))
        # Z-bar operator should not change the state
        new_state = np.dot(self.z_bar, self.initialized_state)
        self.assertEqual(self.initialized_state.shape, new_state.shape)
        self.assertTrue(np.all(self.initialized_state == new_state))


class Test10QSteaneCode(unittest.TestCase):
    """Tests for Steane code functions for the 7+3 qubit system."""

    def test_10q_phase_and_bit_flip_error_correction(self):
        """Test 10-qubit Steane initialization, phase, and bit flip error corrections"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        # -
        initialized_zero_state = initialize_steane_logical_state(ZERO_STATE7Q)
        initialized_zero_state = ancilla_reset(initialized_zero_state, 3)
        # try 5 random phase flips (equal chance of any or no qubits)
        for _ in range(5):
            phase_error_state = phase_flip_error(initialized_zero_state, 10)[0]
            corrected_state = steane_phase_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, 3)
            self.assertTrue(np.allclose(initialized_zero_state, corrected_state))
        # try 5 random bit flips (equal chance of any or no qubits)
        for _ in range(5):
            phase_error_state = bit_flip_error(initialized_zero_state, 10)[0]
            corrected_state = steane_bit_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, 3)
            self.assertTrue(np.allclose(initialized_zero_state, corrected_state))
        # -
        initialized_one_state = initialize_steane_logical_state(ONE_STATE7Q)
        initialized_one_state = ancilla_reset(initialized_one_state, 3)
        # try 5 random phase flips
        for _ in range(5):
            phase_error_state = phase_flip_error(initialized_one_state, 10)[0]
            corrected_state = steane_phase_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, 3)
            self.assertTrue(np.allclose(initialized_one_state, corrected_state))
        # try 5 random bit flips
        for _ in range(5):
            phase_error_state = bit_flip_error(initialized_one_state, 10)[0]
            corrected_state = steane_bit_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, 3)
            self.assertTrue(np.allclose(initialized_one_state, corrected_state))
        # -
        initialized_superpos_state = initialize_steane_logical_state(SUPERPOS_STATE7Q)
        initialized_superpos_state = ancilla_reset(initialized_superpos_state, 3)
        # try 5 random phase flips
        for _ in range(5):
            phase_error_state = phase_flip_error(initialized_superpos_state, 10)[0]
            corrected_state = steane_phase_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, 3)
            self.assertTrue(np.allclose(initialized_superpos_state, corrected_state))
        # try 5 random bit flips
        for _ in range(5):
            phase_error_state = bit_flip_error(initialized_superpos_state, 10)[0]
            corrected_state = steane_bit_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, 3)
            self.assertTrue(np.allclose(initialized_superpos_state, corrected_state))

class Test13QSteaneCode(unittest.TestCase):
    """Tests for Steane code functions for the 7+6 qubit system."""

    def test_13q_phase_and_bit_flip_error_correction(self):
        """Test 13-qubit Steane initialization, phase, and bit flip error corrections"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        n_qubits = 7  # number of data qubits in our system
        n_ancilla = 6 # for "simultaneous" Steane
        n_qtotal = n_qubits + n_ancilla
        random.seed(11)
        # -
        initialized_zero_state = initialize_larger_steane_code(ZERO_STATE7Q)
        self.assertEqual(initialized_zero_state.shape, (2**n_qtotal,))
        initialized_zero_state = ancilla_reset(initialized_zero_state, n_ancilla)
        # too slow for multiple tests - phase flip
        phase_error_state = phase_flip_error(initialized_zero_state, n_qtotal)[0]
        corrected_state = simultaneous_steane_code(phase_error_state)
        corrected_state = ancilla_reset(corrected_state, n_ancilla)
        self.assertTrue(np.allclose(initialized_zero_state, corrected_state))
        # too slow for multiple tests - bit flip
        # too slow for multiple tests - phase and bit flip
        self.assertTrue(
            False, "Check the paper for 13 qubit steane - debugging `initialize_larger_steane_code`"
        )

    # def test_simultaneous_steane_code(self):
    #     """Test `simultaneous_steane_code()`"""
    #     LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
    #     random.seed(10)
    #     n_qubits = 7  # number of data qubits in our system
    #     n_ancilla = 6 # for "simultaneous" Steane
    #     n_qtotal = n_qubits + n_ancilla
    #     initial_state = np.kron(
    #           zero, np.kron(
    #           zero, np.kron(
    #           zero, np.kron(
    #           zero, np.kron(
    #           zero, np.kron(zero, zero))))))
    #     self.assertEqual(initial_state.shape, (2**n_qubits,))
    #     final_vector_state = initialize_larger_steane_code(initial_state)
    #     self.assertEqual(final_vector_state.shape, (2**n_qtotal,)) # pylint: disable=no-member
    #     error_state = phase_flip_error(
    #         bit_flip_error(final_vector_state, n_qtotal)[0], n_qtotal)[0]
    #     self.assertEqual(error_state.shape, (2**n_qtotal,))
    #     corrected_vector_state = simultaneous_steane_code(error_state)
    #     # TODO: add some good tests checking the states


if __name__ == '__main__':
    unittest.main()