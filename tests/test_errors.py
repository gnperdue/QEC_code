'''
Usage:
    python test_errors.py
'''
import unittest
import random
import sys
import logging
import numpy as np
from general_qec.errors import gate_error
from general_qec.errors import errored_adj_CNOT, errored_non_adj_CNOT
from general_qec.errors import random_qubit_x_error, random_qubit_z_error
from general_qec.gates import sigma_y
from general_qec.qec_helpers import one, zero, superpos

LOGGER = logging.getLogger(__name__)


class TestErrors(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module."""

    def test_random_qubit_x_error(self):
        """Test `random_qubit_x_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        logical_state = zero
        errored_logical_state, error_index = random_qubit_x_error(logical_state)
        self.assertEqual(error_index, 0)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(np.all(errored_logical_state == one))
        logical_state = np.kron(one, one)
        errored_logical_state, error_index = random_qubit_x_error(logical_state, (1,1))
        self.assertEqual(error_index, 1)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(np.all(errored_logical_state == np.kron(one, zero)))
        random.seed(11)
        logical_state = np.kron(np.kron(zero, zero), zero)
        errored_logical_state, error_index = random_qubit_x_error(logical_state)
        self.assertEqual(error_index, 2)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(
            np.all(errored_logical_state == np.kron(zero, np.kron(zero, one)))
        )
        logical_state = np.kron(np.kron(zero, zero), zero)
        errored_logical_state, error_index = random_qubit_x_error(logical_state, (2,2))
        self.assertEqual(error_index, 2)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(
            np.all(errored_logical_state == np.kron(zero, np.kron(zero, one)))
        )

    def test_random_qubit_z_error(self):
        """Test `random_qubit_x_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        logical_state = superpos
        errored_logical_state, error_index = random_qubit_z_error(logical_state)
        self.assertEqual(error_index, 0)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertEqual(logical_state[0], errored_logical_state[0])
        self.assertEqual(logical_state[1], -1*errored_logical_state[1])
        logical_state = np.kron(superpos, superpos)
        errored_logical_state, error_index = random_qubit_z_error(logical_state, (1,1))
        self.assertEqual(error_index, 1)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertEqual(logical_state[2], errored_logical_state[2])
        self.assertEqual(logical_state[3], -1*errored_logical_state[3])

    def test_gate_error(self):
        """Test `gate_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(zero, zero)
        rho = np.outer(psi, psi.conj().T)
        rho_prime = gate_error(rho, 0.1, 0, 2)
        self.assertAlmostEqual(np.trace(rho_prime.real), 1) # pylint: disable=no-member
        psi = np.kron(np.matmul(sigma_y, superpos), np.kron(superpos, superpos))
        rho = np.outer(psi, psi.conj().T)
        rho_prime = gate_error(rho, 0.1, 1, 3)
        self.assertAlmostEqual(np.trace(rho_prime.real), 1) # pylint: disable=no-member
        self.assertAlmostEqual(rho_prime[0][0], 1./8 + 0j)

    def test_errored_cnots(self):
        """Test various errored cnot functions"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(one, zero)
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CNOT gate
        errored_rho = errored_adj_CNOT(rho, 0, 1, [0., 0.])
        self.assertAlmostEqual(errored_rho[3][3], 1+0j)
        # apply a high-error CNOT gate
        errored_rho = errored_adj_CNOT(rho, 0, 1, [0., 0.5])
        self.assertAlmostEqual(errored_rho[3][3], 2./3.+0j)
        psi = np.kron(np.kron(one, zero), zero)
        print(psi)
        rho = np.outer(psi, psi.conj().T)
        print(rho)
        # apply a zero-error CNOT gate
        errored_rho = errored_non_adj_CNOT(rho, 0, 2, [0., 0., 0.])
        print(errored_rho)


if __name__ == '__main__':
    unittest.main()
