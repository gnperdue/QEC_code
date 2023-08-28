'''
Usage:
    python test_realistic_codes.py
'''
import unittest
import random

import logging
LOGGER = logging.getLogger(__name__)

import sys
sys.path.append('..')   # the `general_qec` package sits above us

import numpy as np
from general_qec.qec_helpers import one, zero, superpos
from general_qec.qec_helpers import collapse_dm
from circuit_specific.realistic_three_qubit import initialize_three_qubit_realisitc
from circuit_specific.realistic_three_qubit import three_qubit_realistic


class TestRealisticThreeQubit(unittest.TestCase):

    def test_three_qubit_realistic_full(self):
        initial_psi = one # initialize our psi
        # timing parameters
        t1 = 200 * 10**-6 # 200 us
        t2 = 150 * 10**-6 # 150 us
        tg = 20 * 10**-9 # 20 ns
        # probability of gate error for each of five qubits
        p_q0 = 0.0001
        p_q1 = 0.0001
        p_q2 = 0.00001
        p_q3 = 0.0001
        p_q4 = 0.000001
        # state preparation and measurement errors
        spam_prob = 0.00001
        # define your error probability for each qubit
        qubit_error_probs = np.array([p_q0, p_q1, p_q2, p_q3, p_q4])
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1 = t1, t2 = t2, tg = tg,
            qubit_error_probs=qubit_error_probs, spam_prob=spam_prob
        )
        # collapse density matrix when measuring after we initialized our
        # logical state
        collapsed_state = collapse_dm(rho)
        # apply the 3 qubit circuit
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=qubit_error_probs, spam_prob=spam_prob
        )
        # collapse density matrix when measuring after we run the circuit.
        collapsed_state = collapse_dm(rho)
        # TODO - need to test the states...

