"""
This file will contain functions that are useful when implementing logical t1
testing for the 3 qubit code from error models.
"""
import random

import numpy as np
from general_qec.qec_helpers import one, zero
from general_qec.gates import CNOT
from general_qec.errors import *

# Masurement operators for individual qubits
zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
one_meas = np.kron(one, one[np.newaxis].conj().T)


def initialize_three_qubit_realisitc(            # pylint: disable=too-many-arguments
        initial_psi, t1=None, t2=None, tg=None,  # pylint: disable=invalid-name
        qubit_error_probs=None, spam_prob=None
    ):
    """
    Initialize a 3 qubit logical state "realistically" and return the density
    matrix and state.

    * initial_psi: a valid single qubit state vector
    """
    # Initialize the 3 qubit logical state and 2 qubit ancilla pair
    initial_state = np.kron(
        initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero)))
    )

    n = 5 # pylint: disable=invalid-name

    # Find the density matrix of our logical system
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    if spam_prob is not None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)

    # Apply the CNOT gates to the kronecker product of the current 3 qubit
    # density matrix
    if (qubit_error_probs is not None) and \
            ((t1 is not None) and (t2 is not None) and (tg is not None)):
        # first CNOT gate
        rho = prob_line_rad_CNOT(
            initial_rho, 0, 1, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = prob_line_rad_CNOT(
            rho, 1, 2, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
    elif (qubit_error_probs is None) and \
            ((t1 is not None) and (t2 is not None) and (tg is not None)):
        # first CNOT gate
        rho = line_rad_CNOT(
            initial_rho, 0, 1, t1, t2, tg, form = 'rho'
        )
        # second CNOT gate
        rho = line_rad_CNOT(
            rho, 1, 2, t1, t2, tg, form = 'rho'
        )
    elif (qubit_error_probs is not None) and \
            (t1 is None and t2 is None and tg is None):
        # first CNOT gate
        rho = line_errored_CNOT(
            initial_rho, 0, 1, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = line_errored_CNOT(
            rho, 1, 2, qubit_error_probs, form = 'rho'
        )
    else:
        # first CNOT gate
        rho = np.dot(
            CNOT(0, 1, 5), np.dot(initial_rho, CNOT(0, 1, 5).conj().T)
        )
        # second CNOT gate
        rho = np.dot(
            CNOT(1, 2, 5), np.dot(rho, CNOT(1, 2, 5).conj().T)
        )

    return rho


def three_qubit_realistic(
        initial_rho, t1=None, t2=None, tg=None,
        qubit_error_probs=None, spam_prob=None
    ): # pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    """
    Implements the 3 qubit circuit with relaxation and dephasing errors, gate
    error probabilities, and spam errors.

    Outputs the logical state with reset ancilla after correction.

    * initial_rho: initial density matrix of your 5 qubit system
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit in your system
    * spam_prob: The pobability that you have a state prep or measurement error
    Note: times are in seconds.
    """
    nqubits = int(np.log(len(initial_rho))/np.log(2))
    apply_rad_errors = (t1 is not None) and (t2 is not None) and (tg is not None)
    apply_krauss_errors = qubit_error_probs is not None
    apply_spam_errors = spam_prob is not None
    # Define the measurement projection operators
    measure00 = np.kron(np.identity(2**3), np.kron(zero_meas, zero_meas))
    measure01 = np.kron(np.identity(2**3), np.kron(zero_meas, one_meas))
    measure10 = np.kron(np.identity(2**3), np.kron(one_meas, zero_meas))
    measure11 = np.kron(np.identity(2**3), np.kron(one_meas, one_meas))
    all_meas = np.array([measure00, measure01, measure10, measure11])

    # measurement_ops = np.array([
    #     [np.kron(np.identity(2**3), np.kron(zero_meas, zero_meas)),
    #      np.kron(np.identity(2**3), np.kron(zero_meas, one_meas))],
    #     [np.kron(np.identity(2**3), np.kron(one_meas, zero_meas)),
    #      np.kron(np.identity(2**3), np.kron(one_meas, one_meas))]
    # ]).flatten()

    # Apply the CNOT gates needed to change the state of the syndrome ancilla
    detection_rho = initial_rho
    for (control, target) in [(0, 3), (1, 3), (0, 4), (2, 4)]:
        detection_rho = prob_line_rad_CNOT(
            detection_rho, control, target, t1, t2, tg, qubit_error_probs, form='rho')

    # Measure the ancilla qubits
    # -- 1st, apply state measurement error if spam_probs is not empty
    if apply_spam_errors:
        detection_rho = spam_error(detection_rho, spam_prob, 3) # ancilla 0
        detection_rho = spam_error(detection_rho, spam_prob, 4) # ancilla 1

    # -- 2nd, find the probability to measure each case
    # ancilla in 00 -> suggests no errors -> error case00
    # ancilla in 01 -> error on data qubit 2 -> error case01
    # ancilla in 10 -> error on data qubit 1 -> error case10
    # ancilla in 11 -> error on data qubit 0 -> error case11
    error_case01, error_case10, error_case11 = 1, 2, 3
    m00_prob = np.trace(np.dot(measure00.conj().T, np.dot(measure00, detection_rho))).real
    m01_prob = np.trace(np.dot(measure01.conj().T, np.dot(measure01, detection_rho))).real
    m10_prob = np.trace(np.dot(measure10.conj().T, np.dot(measure10, detection_rho))).real
    m11_prob = np.trace(np.dot(measure11.conj().T, np.dot(measure11, detection_rho))).real
    all_probs = np.array([m00_prob, m01_prob, m10_prob, m11_prob])
    assert np.isclose(np.sum(all_probs), 1.0), "Invalid measurement space in three_qubit_realistic"
    # -- 3rd, measure via probability weighted choice
    error_case = random.choices(
        list(range(len(all_probs))), weights=all_probs, k=1
    )[0]
    # -- 4th, measurement collapse of the density matrix
    detection_rho = np.dot(
        all_meas[error_case],
        np.dot(detection_rho, all_meas[error_case].conj().T)
    ) / (all_probs[error_case])

    # Apply correction gates based on ancilla measurements & reset ancilla qubits
    corrected_rho = detection_rho
    # -- 1st, find error index
    error_qubit_index = None
    if error_case == error_case11:
        error_qubit_index = 0
        correction_gate = np.kron(sigma_x, np.identity(2**4))
        corrected_rho = np.dot(
            correction_gate,
            np.dot(corrected_rho, correction_gate.conj().T)
        )
        ancilla_reset_gate = np.kron(np.kron(np.identity(2**3), sigma_x), sigma_x)
        corrected_rho = np.dot(
            ancilla_reset_gate,
            np.dot(corrected_rho, ancilla_reset_gate.conj().T)
        )
        if apply_krauss_errors:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[3], 3, nqubits)
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[4], 4, nqubits)
    elif error_case == error_case10:
        error_qubit_index = 1
        correction_gate = np.kron(np.identity(2), np.kron(sigma_x, np.identity(2**3)))
        corrected_rho = np.dot(
            correction_gate, np.dot(corrected_rho, correction_gate.conj().T)
        )
        ancilla_reset_gate = np.kron(np.kron(np.identity(2**3), sigma_x), np.identity(2))
        corrected_rho = np.dot(
            ancilla_reset_gate,
            np.dot(corrected_rho, ancilla_reset_gate.conj().T)
        )
        if apply_krauss_errors:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[3], 3, nqubits)
    elif error_case == error_case01:
        error_qubit_index = 2
        correction_gate = np.kron(np.identity(2**2), np.kron(sigma_x, np.identity(2**2)))
        corrected_rho = np.dot(
            correction_gate, np.dot(corrected_rho, correction_gate.conj().T)
        )
        ancilla_reset_gate = np.kron(np.identity(2**4), sigma_x)
        corrected_rho = np.dot(
            ancilla_reset_gate,
            np.dot(corrected_rho, ancilla_reset_gate.conj().T)
        )
        if apply_krauss_errors:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[4], 4, nqubits)
    # -- 2nd, apply Krauss errors for state correction if appropriate
    if (error_qubit_index is not None) and apply_krauss_errors:
        corrected_rho = gate_error(
            corrected_rho, qubit_error_probs[error_qubit_index], error_qubit_index, nqubits
        )
    # -- 3rd, apply RAD errors -> assume data and ancilla qubit operations are in paralllel
    if (error_qubit_index is not None) and apply_rad_errors:
        corrected_rho = rad_error(corrected_rho, t1, t2, tg)

    return corrected_rho


def three_qubit_realistic_santi(
        initial_rho, t1=None, t2=None, tg=None,
        qubit_error_probs = None, spam_prob = None
    ): # pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    """
    Implements the 3 qubit circuit with relaxation and dephasing errors, gate
    error probabilities, and spam errors.

    Outputs the logical state with reset ancilla after correction.

    * initial_rho: initial density matrix of your 5 qubit system
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit in your system
    * spam_prob: The pobability that you have a state prep or measurement error
    Note: times are in seconds.

    GP - I think this function makes an inappropriate measurement step at the end.
    It cannot functon correctly if there are no RAD errors.
    """
    # total number of qubits in our system
    n = int(np.log(len(initial_rho))/np.log(2)) # pylint: disable=invalid-name

    # Apply CNOT gates depending on error parameters
    if ((t1 is not None) and (t2 is not None) and (tg is not None)) and \
            (qubit_error_probs is not None):
        # Apply the CNOT gates needed to change the state of the syndrome ancilla
        detection_rho = prob_line_rad_CNOT(
            initial_rho, 0, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        detection_rho = prob_line_rad_CNOT(
            detection_rho, 1, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        detection_rho = prob_line_rad_CNOT(
            detection_rho, 0, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        detection_rho = prob_line_rad_CNOT(
            detection_rho, 2, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1 is not None) and (t2 is not None) and (tg is not None)) and \
            (qubit_error_probs is None):
        # Apply the CNOT gates needed to change the state of the syndrome ancilla
        detection_rho = line_rad_CNOT(
            initial_rho, 0, 3, t1, t2, tg, form = 'rho')
        detection_rho = line_rad_CNOT(
            detection_rho, 1, 3, t1, t2, tg, form = 'rho')
        detection_rho = line_rad_CNOT(
            detection_rho, 0, 4, t1, t2, tg, form = 'rho')
        detection_rho = line_rad_CNOT(
            detection_rho, 2, 4, t1, t2, tg, form = 'rho')
    elif ((t1 is None) and (t2 is None) and (tg is None)) and \
            (qubit_error_probs is not None):
        # Apply the CNOT gates needed to change the state of the syndrome ancilla
        detection_rho = line_errored_CNOT(
            initial_rho, 0, 3, qubit_error_probs, form = 'rho')
        detection_rho = line_errored_CNOT(
            detection_rho, 1, 3, qubit_error_probs, form = 'rho')
        detection_rho = line_errored_CNOT(
            detection_rho, 0, 4, qubit_error_probs, form = 'rho')
        detection_rho = line_errored_CNOT(
            detection_rho, 2, 4, qubit_error_probs, form = 'rho')
    else:
        # Apply the CNOT gates needed to change the state of the syndrome ancilla
        gate1 = CNOT(0, 3, 5)
        gate2 = CNOT(1, 3, 5)
        gate3 = CNOT(0, 4, 5)
        gate4 = CNOT(2, 4, 5)

        detection_rho = np.dot(gate1, np.dot(initial_rho, gate1.conj().T))
        detection_rho = np.dot(gate2, np.dot(detection_rho, gate2.conj().T))
        detection_rho = np.dot(gate3, np.dot(detection_rho, gate3.conj().T))
        detection_rho = np.dot(gate4, np.dot(detection_rho, gate4.conj().T))

    # Apply state measurement error if spam_probs is not empty
    if spam_prob is not None:
        detection_rho = spam_error(detection_rho, spam_prob, 3) # ancilla 0
        detection_rho = spam_error(detection_rho, spam_prob, 4) # ancilla 1

    # Define the measurement projection operators
    M1 = np.kron(np.identity(2**3), np.kron(zero_meas, zero_meas)) # pylint: disable=invalid-name
    M2 = np.kron(np.identity(2**3), np.kron(zero_meas, one_meas))  # pylint: disable=invalid-name
    M3 = np.kron(np.identity(2**3), np.kron(one_meas, zero_meas))  # pylint: disable=invalid-name
    M4 = np.kron(np.identity(2**3), np.kron(one_meas, one_meas))   # pylint: disable=invalid-name
    all_meas = np.array([M1, M2, M3, M4])

    # find the probability to measure each case
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, detection_rho))).real
    m2_prob = np.trace(np.dot(M2.conj().T, np.dot(M2, detection_rho))).real
    m3_prob = np.trace(np.dot(M3.conj().T, np.dot(M3, detection_rho))).real
    m4_prob = np.trace(np.dot(M4.conj().T, np.dot(M4, detection_rho))).real
    all_probs = np.array([m1_prob, m2_prob, m3_prob, m4_prob])

    # find which measurement operator is measured based on their probabilities
    # GP: pretty sure these two lines are bugged - will not work for the case
    # -- where all meas probs are the same, e.g. [0.25,0.25,0.25,0.25] - you
    # -- will always get index 0, and it should be an even pick across the
    # -- values.
    # index = random.choices(all_probs, weights=all_probs, k=1)
    # index = np.where(all_probs == index)[0][0]
    index = random.choices(
        list(range(len(all_probs))), weights=all_probs, k=1
    )[0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(
        all_meas[index],
        np.dot(detection_rho, all_meas[index].conj().T)
    )/(all_probs[index])

    # apply rad error due to time taken to collapse
    if ((t1 is not None) and (t2 is not None) and (tg is not None)):
        rho_prime = rad_error(rho_prime, t1, t2, tg)

    # Create our new density matrix after collapsing ancilla qubits
    detection_rho = rho_prime

    # Initialize error index

    if index == 3: # Error on qubit 0
        error_index = 0
    elif index == 2: # Error on qubit 1
        error_index = 1
    elif index == 1: # Error on qubit 2
        error_index = 2
    elif index == 0:
        error_index = -1

    # apply the error correction based on the detected index
    if error_index == 0: # Error on qubit 0
        correction_gate = np.kron(sigma_x, np.identity(2**4))
        corrected_rho = np.dot(
            correction_gate,
            np.dot(detection_rho, correction_gate.conj().T)
        )
        if qubit_error_probs is not None:
            # gate error probability
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[0], 0, n)
        if ((t1 is not None) and (t2 is not None) and (tg is not None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 1: # Error on qubit 1
        correction_gate = np.kron(
            np.identity(2), np.kron(sigma_x, np.identity(2**3))
        )
        corrected_rho = np.dot(
            correction_gate, np.dot(detection_rho, correction_gate.conj().T)
        )
        if qubit_error_probs is not None:
            # gate error probability
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[1], 1, n)
        if ((t1 is not None) and (t2 is not None) and (tg is not None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 2: # Error on qubit 2
        correction_gate = np.kron(
            np.identity(2**2), np.kron(sigma_x, np.identity(2**2))
        )
        corrected_rho = np.dot(
            correction_gate, np.dot(detection_rho, correction_gate.conj().T)
        )
        if qubit_error_probs is not None:
            # gate error probability
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[2], 2, n)
        if ((t1 is not None) and (t2 is not None) and (tg is not None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    else: # No error occured
        corrected_rho = detection_rho
        if ((t1 is not None) and (t2 is not None) and (tg is not None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    # Reset the ancillas by projecting to the |00><00| basis
    # apply correct measurement collapse of the density matrix
    #
    # GP - this is a redundant "measurement" and will fail if there are no RAD
    # errors (it is possible to have very valid states where projection to M1
    # will have zero overlap - but RAD errors ensure you always have some non-zero
    # probability there)
    ancilla_reset_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, corrected_rho)))
    reset_rho = np.dot(M1, np.dot(corrected_rho, M1.conj().T))/(ancilla_reset_prob)

    return reset_rho
