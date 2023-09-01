'''
Usage:
    python test_errors_rad.py

Test `errors.py` - focus on the relaxation and dephasing ("rad") functions.
'''
import unittest
import random
import sys
import logging
import numpy as np
from general_qec.qec_helpers import zero, superpos
from general_qec.errors import rad_error
from general_qec.qec_helpers import collapse_dm

LOGGER = logging.getLogger(__name__)


class TestRadErrors(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module -- the "rad" functions."""

    def test_rad_error(self):
        """Test `rad_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # basic check
        psi = np.kron(zero, zero)
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 0.1, 0.1, 1e-9)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        # more intersting state, larger decay
        psi = 1./np.sqrt(2)*np.array([1.0, 0.0, 0.0, 1.0])
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 1e-8, 1e-8, 1e-9)
        random.seed(10)  # carefully chosen for collapsed state
        psi_prime = collapse_dm(rho_prime)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        self.assertTrue(np.all(psi_prime == np.array([0., 0., 1., 0.])),
                        msg="Random seed may not have produced the right collapsed state.")
        # more intersting state, just t1 -> drive hard to ground state
        psi = 1./np.sqrt(2)*np.array([1.0, 0.0, 0.0, 1.0])
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 1, 1e9, 1e2)
        psi_prime = collapse_dm(rho_prime)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        self.assertTrue(np.all(psi_prime == np.array([1., 0., 0., 0.])),
                        msg="Random seed may not have produced the right collapsed state.")
        # more intersting state, just t2 -> drive off-diagonals to zero hard
        psi = np.kron(superpos, superpos)
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 1e9, 1, 1e2)
        random.seed(10)  # carefully chosen for collapsed state
        psi_prime = collapse_dm(rho_prime)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        self.assertAlmostEqual(np.sum(rho_prime) - np.trace(rho_prime), 0.0)
        self.assertTrue(np.all(psi_prime == np.array([0., 0., 1., 0.])),
                        msg="Random seed may not have produced the right collapsed state.")


if __name__ == '__main__':
    unittest.main()
