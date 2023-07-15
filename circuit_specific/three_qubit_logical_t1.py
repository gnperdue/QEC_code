# This file will contain functions that are useful when implementing logical t1 testing for the 3 qubit code from error models

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *

### Implements the 3 qubit circuit with just relaxation and dephasing errors 
### Outputs the logical state with reset ancilla after correction
def three_qubit_rad(initial_rho, t1, t2, tg):
    # apply a random x error using the density matrix
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,2)
    
    # Apply the error to the qubit (no error may occur)
    if error_index == 0:
        gate = np.kron(sigma_x, np.identity(2**4))
        errored_rho = np.dot(gate, np.dot(initial_rho, gate.conj().T))

    elif error_index == 1:
        gate = np.kron(np.kron(np.identity(2), sigma_x), np.identity(2**3))
        errored_rho = np.dot(gate, np.dot(initial_rho, gate.conj().T))

    elif error_index == 2:
        gate = np.kron(np.identity(2**2), np.kron(sigma_x, np.identity(2**2)))
        errored_rho = np.dot(gate, np.dot(initial_rho, gate.conj().T))

    else:
        errored_rho = initial_rho

    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    prob_sum = 0
    state_probs = np.array([])
    for i in range(len(errored_rho)):
        prob_sum += np.abs(errored_rho[i,i])
        state_probs = np.append(state_probs, errored_rho[i,i])

    # Apply the CNOT gates needed to change the state of the syndrome ancilla 
    detection_rho = line_rad_CNOT(errored_rho, 0, 3, t1, t2, tg, form = 'rho')
    detection_rho = line_rad_CNOT(detection_rho, 1, 3, t1, t2, tg, form = 'rho')
    detection_rho = line_rad_CNOT(detection_rho, 0, 4, t1, t2, tg, form = 'rho')
    detection_rho = line_rad_CNOT(detection_rho, 2, 4, t1, t2, tg, form = 'rho')

    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    prob_sum = 0
    state_probs = np.array([])
    for i in range(len(detection_rho)):
        prob_sum += np.abs(detection_rho[i,i])
        state_probs = np.append(state_probs, detection_rho[i,i])
    
    # Measure the ancilla qubits and collapse them
    collapsed_state = collapse_ancilla(np.sqrt(state_probs), 2)
    logical_bits = vector_state_to_bit_state(collapsed_state, 5)[0][0]
    
    # Create our new density matrix after collapsing ancilla qubits
    detection_rho = np.kron(collapsed_state, collapsed_state[np.newaxis].conj().T)
    
    # Initialize error index
    error_index = -1

    if ((logical_bits[3] == '1') and (logical_bits[4] == '1')): # Error on qubit 0
        error_index = 0
    elif ((logical_bits[3] == '1') and (logical_bits[4] == '0')): # Error on qubit 1
        error_index = 1
    elif ((logical_bits[3] == '0') and (logical_bits[4] == '1')): # Error on qubit 2
        error_index = 2


    if error_index == 0: # Error on qubit 0
        correction_gate = np.kron(sigma_x, np.identity(2**4))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 1: # Error on qubit 1
        correction_gate = np.kron(np.identity(2), np.kron(sigma_x, np.identity(2**3)))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error


    elif error_index == 2: # Error on qubit 2
        correction_gate = np.kron(np.identity(2**2), np.kron(sigma_x, np.identity(2**2)))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error


    else: # No error occured
        corrected_rho = detection_rho
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    prob_sum = 0
    state_probs = np.array([])
    for i in range(len(corrected_rho)):
        prob_sum += np.abs(corrected_rho[i,i])
        state_probs = np.append(state_probs, corrected_rho[i,i])

    # reset ancilla qubits (since we have a probability of being in a state we squareroot it to find the magnitude)
    reset_state = np.sqrt(ancilla_reset(state_probs, 2))

    # now we find the density matrix of our state 
    reset_rho = np.kron(reset_state, reset_state[np.newaxis].conj().T)
    
    return reset_rho, prob_sum


### Implements the 3 qubit circuit with relaxation and dephasing errors and gate error probabilities. 
### Outputs the logical state with reset ancilla after correction
def three_qubit_realistic(initial_rho, t1, t2, tg, qubit_error_probs):

    # total number of qubits in our system
    n = int(np.log(len(initial_rho))/np.log(2))

    # Apply the CNOT gates needed to change the state of the syndrome ancilla 
    detection_rho = prob_line_rad_CNOT(initial_rho, 0, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
    detection_rho = prob_line_rad_CNOT(detection_rho, 1, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
    detection_rho = prob_line_rad_CNOT(detection_rho, 0, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
    detection_rho = prob_line_rad_CNOT(detection_rho, 2, 4, t1, t2, tg, qubit_error_probs, form = 'rho')

    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    prob_sum = 0
    state_probs = np.array([])
    for i in range(len(detection_rho)):
        prob_sum += np.abs(detection_rho[i,i])
        state_probs = np.append(state_probs, detection_rho[i,i])
    
    # Measure the ancilla qubits and collapse them
    collapsed_state = collapse_ancilla(np.sqrt(state_probs), 2)
    logical_bits = vector_state_to_bit_state(collapsed_state, 5)[0][0]
    
    # Create our new density matrix after collapsing ancilla qubits
    detection_rho = np.kron(collapsed_state, collapsed_state[np.newaxis].conj().T)
    
    # Initialize error index
    error_index = -1

    if ((logical_bits[3] == '1') and (logical_bits[4] == '1')): # Error on qubit 0
        error_index = 0
    elif ((logical_bits[3] == '1') and (logical_bits[4] == '0')): # Error on qubit 1
        error_index = 1
    elif ((logical_bits[3] == '0') and (logical_bits[4] == '1')): # Error on qubit 2
        error_index = 2

    # apply the error correction based on the detected index
    if error_index == 0: # Error on qubit 0
        correction_gate = np.kron(sigma_x, np.identity(2**4))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        
        corrected_rho = gate_error(corrected_rho, qubit_error_probs[0], 0, n) # gate error probability
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 1: # Error on qubit 1
        correction_gate = np.kron(np.identity(2), np.kron(sigma_x, np.identity(2**3)))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        
        corrected_rho = gate_error(corrected_rho, qubit_error_probs[1], 1, n) # gate error probability
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 2: # Error on qubit 2
        correction_gate = np.kron(np.identity(2**2), np.kron(sigma_x, np.identity(2**2)))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        
        corrected_rho = gate_error(corrected_rho, qubit_error_probs[2], 2, n) # gate error probability
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    else: # No error occured
        corrected_rho = detection_rho
        corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    prob_sum = 0
    state_probs = np.array([])
    for i in range(len(corrected_rho)):
        prob_sum += np.abs(corrected_rho[i,i])
        state_probs = np.append(state_probs, corrected_rho[i,i])

    # reset ancilla qubits (since we have a probability of being in a state we squareroot it to find the magnitude)
    reset_state = np.sqrt(ancilla_reset(state_probs, 2))

    # now we find the density matrix of our state 
    reset_rho = np.kron(reset_state, reset_state[np.newaxis].conj().T)
    
    return reset_rho, prob_sum