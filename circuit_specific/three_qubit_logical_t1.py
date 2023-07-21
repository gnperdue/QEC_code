# This file will contain functions that are useful when implementing logical t1 testing for the 3 qubit code from error models

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *

# Masurement operators for individual qubits
zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
one_meas = np.kron(one, one[np.newaxis].conj().T)

### Initialized your 3 qubit logical state realistically and return the density matrix and state
def initialize_three_qubit_realisitc(initial_psi, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):    
    # Initialize the 3 qubit logical state
    initial_state = np.kron(initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero))))
    
    n = 5
    
    # Find the density matrix of our logical system
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    if spam_prob != None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)
    
    if (qubit_error_probs is not None) and (t1!=None and t2!=None and tg!=None):
        # Apply the CNOT gates to the kronecker product of the current 3 qubit density matrix
        rho = prob_line_rad_CNOT(initial_rho, 0, 1, t1, t2, tg, qubit_error_probs, form = 'rho') # first CNOT gate
        rho = prob_line_rad_CNOT(rho, 1, 2, t1, t2, tg, qubit_error_probs, form = 'rho') # second CNOT gate
    elif (qubit_error_probs is None) and (t1!=None and t2!=None and tg!=None):
        # Apply the CNOT gates to the kronecker product of the current 3 qubit density matrix
        rho = line_rad_CNOT(initial_rho, 0, 1, t1, t2, tg, form = 'rho') # first CNOT gate
        rho = line_rad_CNOT(rho, 1, 2, t1, t2, tg, form = 'rho') # second CNOT gate
    elif (qubit_error_probs is not None) and (t1==None and t2==None and tg==None):
        # Apply the CNOT gates to the kronecker product of the current 3 qubit density matrix
        rho = line_errored_CNOT(initial_rho, 0, 1, qubit_error_probs, form = 'rho') # first CNOT gate
        rho = line_errored_CNOT(rho, 1, 2, qubit_error_probs, form = 'rho') # second CNOT gate
    else:
        # Apply the CNOT gates to the kronecker product of the current 3 qubit density matrix
        rho = np.dot(CNOT(0, 1, 5), np.dot(initial_rho, CNOT(0, 1, 5).conj().T)) # first CNOT gate
        rho = np.dot(CNOT(1, 2, 5), np.dot(rho, CNOT(1, 2, 5).conj().T)) # second CNOT gate
    
    return rho



### Implements the 3 qubit circuit with relaxation and dephasing errors, gate error probabilities, and spam errors. 
### Outputs the logical state with reset ancilla after correction
def three_qubit_realistic(initial_rho, t1 = None, t2 = None, tg = None, qubit_error_probs = None, spam_prob = None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # total number of qubits in our system
    n = int(np.log(len(initial_rho))/np.log(2))
    
    # Apply state prep error if spam_probs is not empty
    if spam_prob != None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)
    
    # Apply CNOT gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # Apply the CNOT gates needed to change the state of the syndrome ancilla 
        detection_rho = prob_line_rad_CNOT(initial_rho, 0, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        detection_rho = prob_line_rad_CNOT(detection_rho, 1, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        detection_rho = prob_line_rad_CNOT(detection_rho, 0, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        detection_rho = prob_line_rad_CNOT(detection_rho, 2, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # Apply the CNOT gates needed to change the state of the syndrome ancilla 
        detection_rho = line_rad_CNOT(initial_rho, 0, 3, t1, t2, tg, form = 'rho')
        detection_rho = line_rad_CNOT(detection_rho, 1, 3, t1, t2, tg, form = 'rho')
        detection_rho = line_rad_CNOT(detection_rho, 0, 4, t1, t2, tg, form = 'rho')
        detection_rho = line_rad_CNOT(detection_rho, 2, 4, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # Apply the CNOT gates needed to change the state of the syndrome ancilla 
        detection_rho = line_errored_CNOT(initial_rho, 0, 3, qubit_error_probs, form = 'rho')
        detection_rho = line_errored_CNOT(detection_rho, 1, 3, qubit_error_probs, form = 'rho')
        detection_rho = line_errored_CNOT(detection_rho, 0, 4, qubit_error_probs, form = 'rho')
        detection_rho = line_errored_CNOT(detection_rho, 2, 4, qubit_error_probs, form = 'rho')
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
    if spam_prob != None:
        detection_rho = spam_error(detection_rho, spam_prob, 3) # ancilla 0
        detection_rho = spam_error(detection_rho, spam_prob, 4) # ancilla 1


    # Define the measurement projection operators
    M1 = np.kron(np.identity(2**3), np.kron(zero_meas, zero_meas))
    M2 = np.kron(np.identity(2**3), np.kron(zero_meas, one_meas))
    M3 = np.kron(np.identity(2**3), np.kron(one_meas, zero_meas))
    M4 = np.kron(np.identity(2**3), np.kron(one_meas, one_meas))
    
    all_meas = np.array([M1, M2, M3, M4])
    
    # find the probability to measure each case
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, detection_rho)))
    m2_prob = np.trace(np.dot(M2.conj().T, np.dot(M2, detection_rho)))
    m3_prob = np.trace(np.dot(M3.conj().T, np.dot(M3, detection_rho)))
    m4_prob = np.trace(np.dot(M4.conj().T, np.dot(M4, detection_rho)))
    
    all_probs = np.array([m1_prob, m2_prob, m3_prob, m4_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]
    
    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(detection_rho, all_meas[index].conj().T))/(all_probs[index])
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
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        
        if qubit_error_probs is not None:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[0], 0, n) # gate error probability
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 1: # Error on qubit 1
        correction_gate = np.kron(np.identity(2), np.kron(sigma_x, np.identity(2**3)))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        
        if qubit_error_probs is not None:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[1], 1, n) # gate error probability
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    elif error_index == 2: # Error on qubit 2
        correction_gate = np.kron(np.identity(2**2), np.kron(sigma_x, np.identity(2**2)))
        corrected_rho = np.dot(correction_gate, np.dot(detection_rho, correction_gate.conj().T))
        
        if qubit_error_probs is not None:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[2], 2, n) # gate error probability
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    else: # No error occured
        corrected_rho = detection_rho
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            corrected_rho = rad_error(corrected_rho, t1, t2, tg) # apply rad error

    # Reset the ancillas by projecting to the |00><00| basis    
    # apply correct measurement collapse of the density matrix
    ancilla_reset_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, corrected_rho)))
    reset_rho = np.dot(M1, np.dot(corrected_rho, M1.conj().T))/(ancilla_reset_prob)


    return reset_rho

