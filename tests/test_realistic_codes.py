'''
Usage:
    python test_realistic_codes.py
'''
import unittest
import logging
import random
import sys
import numpy as np
from general_qec.errors import random_qubit_x_error
from general_qec.qec_helpers import one
from general_qec.qec_helpers import collapse_dm
from circuit_specific.realistic_three_qubit import initialize_three_qubit_realisitc
from circuit_specific.realistic_three_qubit import three_qubit_realistic

LOGGER = logging.getLogger(__name__)


class TestRealisticThreeQubit(unittest.TestCase):
    """Tests for the `realistic_three_qubit` module."""

    def test_three_qubit_realistic_full(self):
        """Pseudo-functional test of `three_qubit_realistic_full()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        initial_psi = one # initialize our psi
        # timing parameters in microseconds
        t1 = 200 * 10**-6 # pylint: disable=invalid-name
        t2 = 150 * 10**-6 # pylint: disable=invalid-name
        tg = 20 * 10**-9  # pylint: disable=invalid-name
        # probability of gate error for each of five qubits
        # -- these are tiny errors and low likelihood
        p_q0 = 0.0001
        p_q1 = 0.0001
        p_q2 = 0.00001
        p_q3 = 0.0001
        p_q4 = 0.000001
        # state preparation and measurement errors
        spam_prob = 0.00001
        # define your error probability for each qubit
        qubit_error_probs = np.array([p_q0, p_q1, p_q2, p_q3, p_q4])
        # TODO: test more initalization cases... (Nones, etc.)
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=qubit_error_probs, spam_prob=spam_prob
        )
        # collapse density matrix when measuring after we initialized our
        # logical state
        random.seed(10)
        collapsed_state = collapse_dm(rho)
        self.assertAlmostEqual(collapsed_state[28], 1+0j)
        # apply the 3 qubit circuit
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=qubit_error_probs, spam_prob=spam_prob
        )
        # collapse density matrix when measuring after we run the circuit.
        random.seed(10)
        collapsed_state = collapse_dm(rho)
        self.assertAlmostEqual(collapsed_state[28], 1+0j)
        # throw an x error on the data qubits
        errored_state, _ = random_qubit_x_error(collapsed_state, (1,1))
        self.assertAlmostEqual(errored_state[28], 0+0j)
        rho = np.outer(errored_state, errored_state.conj().T)
        # repair it
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=qubit_error_probs, spam_prob=spam_prob
        )
        random.seed(10)
        collapsed_state = collapse_dm(rho)
        self.assertAlmostEqual(collapsed_state[28], 1+0j)


if __name__ == '__main__':
    unittest.main()
