# This file will contain functions that are useful when implementing logical t1 testing for the Steane code from error models

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *

### implemenent the Steane code with depolarization, rad, and spam errors
def realistic_steane(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None, info = True):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # info: Do you want to print out debugging/helpful info?

    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    # Apply state prep error if spam_probs is not empty
    if spam_prob != None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)
    
    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    state_probs = np.array([])
    tot = 0
    for i in range(len(initial_rho)):
        tot += np.abs(initial_rho[i, i])
        state_probs = np.append(state_probs, initial_rho[i,i])
    
    initial_state = np.sqrt(state_probs)
    
    if info:
        print('Initial state before Z correction:')
        print_state_info(initial_state, 10)
        print(' - ')

    ### Implements the 7 Qubit Steane phase correction code using line connectivity
    if info:
        print('Applying Stabilizer operators 1-3.')
        print('...')
    # - - - - - - - - - - # Z Error Correction # - - - - - - - - - - #
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    current_rho = np.dot(ancilla_hadamard, np.dot(initial_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() == None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
        
    
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() != None):
        # apply K1 first:
        current_rho = prob_line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 7, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 7, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 7, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = prob_line_rad_CNOT(current_rho, 8, 0, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = prob_line_rad_CNOT(current_rho, 9, 1, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() == None):
        # apply K1 first:
        current_rho = line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 7, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 7, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 7, 6, t1, t2, tg, form = 'rho')
        # apply K2:
        current_rho = line_rad_CNOT(current_rho, 8, 0, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, form = 'rho')
        # apply K3:
        current_rho = line_rad_CNOT(current_rho, 9, 1, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 6, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs.all() != None):
        # apply K1 first:
        current_rho = line_errored_CNOT(current_rho, 7, 3, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 7, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 7, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 7, 6, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = line_errored_CNOT(current_rho, 8, 0, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 6, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = line_errored_CNOT(current_rho, 9, 1, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 6, qubit_error_probs, form = 'rho')
    else:
        # Define the Stabilizer Operators as CNOT gates between line adjacent qubits 
        K1 = np.dot(CNOT(7, 3, 10), np.dot(CNOT(7, 4, 10), np.dot(CNOT(7, 5, 10), CNOT(7, 6, 10))))
        K2 = np.dot(CNOT(8, 0, 10), np.dot(CNOT(8, 2, 10), np.dot(CNOT(8, 4, 10), CNOT(8, 6, 10))))
        K3 = np.dot(CNOT(9, 1, 10), np.dot(CNOT(9, 2, 10), np.dot(CNOT(9, 5, 10), CNOT(9, 6, 10))))
        current_rho = np.dot(K1, np.dot(current_rho, K1.conj().T))
        current_rho = np.dot(K2, np.dot(current_rho, K2.conj().T))
        current_rho = np.dot(K3, np.dot(current_rho, K3.conj().T))
    
    
    # apply the second hadamard to the ancillas
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() == None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 7) # ancilla 0
        current_rho = spam_error(current_rho, spam_prob, 8) # ancilla 1
        current_rho = spam_error(current_rho, spam_prob, 9) # ancilla 2

    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    state_probs = np.array([])
    tot = 0
    for i in range(len(current_rho)):
        tot += np.abs(current_rho[i, i])
        state_probs = np.append(state_probs, current_rho[i,i])

    # Measure the ancilla qubits and collapse them
    collapsed_state = collapse_ancilla(np.sqrt(state_probs), 3)
    if info:
        print('Collapsed state after ancilla measurement:')
        print_state_info(collapsed_state, 10)
        print(' - ')

    # Create our new density matrix after collapsing ancilla qubits
    rho = np.kron(collapsed_state, collapsed_state[np.newaxis].conj().T)
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho, t1, t2, tg)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(collapsed_state))/np.log(2))

    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, n)[0][0]

    # find index
    m_one = 0
    m_two = 0
    m_three = 0
    if bits[7] == '1':
        m_one = 1
    if bits[8] == '1':
        m_two = 1
    if bits[9] == '1':
        m_three = 1

    # Which qubit do we perform the Z gate on
    index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_rho = rho

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        current_rho = np.dot(operation, np.dot(rho, operation.conj().T))
        
        if qubit_error_probs.all() != None:
            current_rho = gate_error(current_rho, qubit_error_probs[index], index, n) # depolarizing error
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            current_rho = rad_error(current_rho, t1, t2, tg) # apply an error for correction gate time

    final_rho = current_rho
    
    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    state_probs = np.array([])
    tot = 0
    for i in range(len(final_rho)):
        tot += np.abs(final_rho[i,i])
        state_probs = np.append(state_probs, final_rho[i,i])

    final_state_z = np.sqrt(state_probs)
    if info:
        print('Final state after Z correction:')
        print_state_info(final_state_z, 10)
        print('- - -')

    # Reset the ancilla qubits:
    initial_state = ancilla_reset(final_state_z, 3)
    if info:
        print('Initial state before X correction:')
        print_state_info(initial_state, 10)
        print(' - ')

    ### Implements the 7 Qubit Steane bit correction code using line connectivity
    if info:
        print('Applying Stabilizer operators 4-6.')
        print('...')
    
    # Create our new density matrix with reset ancillas
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    # Apply state prep error if spam_probs is not empty
    if spam_prob != None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)
    
    # - - - - - - - - - - # X Error Correction # - - - - - - - - - - #

    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() == None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
        
    
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() != None):
        # apply K1 first:
        current_rho = prob_line_rad_CZ(current_rho, 7, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 7, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 7, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 7, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = prob_line_rad_CZ(current_rho, 8, 0, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = prob_line_rad_CZ(current_rho, 9, 1, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() == None):
        # apply K1 first:
        current_rho = line_rad_CZ(current_rho, 7, 3, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 7, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 7, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 7, 6, t1, t2, tg, form = 'rho')
        # apply K2:
        current_rho = line_rad_CZ(current_rho, 8, 0, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 6, t1, t2, tg, form = 'rho')
        # apply K3:
        current_rho = line_rad_CZ(current_rho, 9, 1, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 6, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs.all() != None):
        # apply K1 first:
        current_rho = line_errored_CZ(current_rho, 7, 3, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 7, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 7, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 7, 6, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = line_errored_CZ(current_rho, 8, 0, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 6, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = line_errored_CZ(current_rho, 9, 1, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 6, qubit_error_probs, form = 'rho')
    else:
        # Define the Stabilizer Operators as CNOT gates between line adjacent qubits 
        K4 = np.dot(CZ(7, 3, 10), np.dot(CZ(7, 4, 10), np.dot(CZ(7, 5, 10), CZ(7, 6, 10))))
        K5 =np.dot(CZ(8, 0, 10), np.dot(CZ(8, 2, 10), np.dot(CZ(8, 4, 10), CZ(8, 6, 10))))
        K6 =np.dot(CZ(9, 1, 10), np.dot(CZ(9, 2, 10), np.dot(CZ(9, 5, 10), CZ(9, 6, 10)))) 
        
        current_rho = np.dot(K4, np.dot(current_rho, K4.conj().T))
        current_rho = np.dot(K5, np.dot(current_rho, K5.conj().T))
        current_rho = np.dot(K6, np.dot(current_rho, K6.conj().T))
    
    
    # apply the second hadamard to the ancillas
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs.all() == None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs.all() != None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 7) # ancilla 0
        current_rho = spam_error(current_rho, spam_prob, 8) # ancilla 1
        current_rho = spam_error(current_rho, spam_prob, 9) # ancilla 2
    
    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    state_probs = np.array([])
    tot = 0
    for i in range(len(current_rho)):
        tot += np.abs(current_rho[i, i])
        state_probs = np.append(state_probs, current_rho[i,i])

    # Measure the ancilla qubits and collapse them
    collapsed_state = collapse_ancilla(np.sqrt(state_probs), 3)
    if info:
        print('Collapsed state after ancilla measurement:')
        print_state_info(collapsed_state, 10)
        print(' - ')

    # Create our new density matrix after collapsing ancilla qubits
    rho = np.kron(collapsed_state, collapsed_state[np.newaxis].conj().T)

    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(collapsed_state))/np.log(2))

    # Measure the three ancilla qubits
    # Applying the X gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, n)[0][0]

    # find index
    m_four = 0
    m_five = 0
    m_six = 0
    if bits[7] == '1':
        m_four = 1
    if bits[8] == '1':
        m_five = 1
    if bits[9] == '1':
        m_six = 1

    # Which qubit do we perform the Z gate on
    index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_rho = rho

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        current_rho = np.dot(operation, np.dot(rho, operation.conj().T))
        
        if qubit_error_probs.all() != None:
            current_rho = gate_error(current_rho, qubit_error_probs[index], index, n) # depolarizing error
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            current_rho = rad_error(current_rho, t1, t2, tg) # apply an error for correction gate time

    final_rho = current_rho
    
    # probability of the state measurments from the density matrix are defined as Tr(p*rho)
    state_probs = np.array([])
    tot = 0
    for i in range(len(final_rho)):
        tot += np.abs(final_rho[i,i])
        state_probs = np.append(state_probs, final_rho[i,i])

    final_state_x = np.sqrt(state_probs)
    if info:
        print('Final state after X correction:')
        print_state_info(final_state_x, 10)

    final_state = final_state_x
    
    return final_rho, final_state

# ### implemenent the Steane code with rad errors
# def rad_steane(initial_rho, t1, t2, tg, info = True):
#     # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
#     # t1: The relaxation time of each physical qubit in your system
#     # t2: The dephasing time of each physical qubit in your system
#     # tg: The gate time of your gate operations 
#     # info: Do you want to print out debugging/helpful info?
    
#     # probability of the state measurments from the density matrix are defined as Tr(p*rho)
#     state_probs = np.array([])
#     tot = 0
#     for i in range(len(initial_rho)):
#         tot += np.abs(initial_rho[i, i])
#         state_probs = np.append(state_probs, initial_rho[i,i])
    
#     initial_state = np.sqrt(state_probs)
    
#     if info:
#         print('Initial state before Z correction:')
#         print_state_info(initial_state, 10)
#         print(' - ')

#     ### Implements the 7 Qubit Steane phase correction code using line connectivity
#     if info:
#         print('Applying Stabilizer operators 1-3.')
#         print('...')
#     # - - - - - - - - - - # Z Error Correction # - - - - - - - - - - #
#     # apply the first hadamard to the ancillas
#     ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
#     current_rho = np.dot(ancilla_hadamard, np.dot(initial_rho, ancilla_hadamard.conj().T))
#     current_rho = rad_error(current_rho, t1, t2, tg)

#     # normalize current_rho after applying the hadamard gate
# #     current_rho = 8 * current_rho
    
#     # apply the control stabilizer gates to current_rho

#     # apply K1 first:
#     current_rho = line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 7, 4, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 7, 5, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 7, 6, t1, t2, tg, form = 'rho')

#     # apply K2:
#     current_rho = line_rad_CNOT(current_rho, 8, 0, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, form = 'rho')

#     # apply K3:
#     current_rho = line_rad_CNOT(current_rho, 9, 1, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 9, 2, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CNOT(current_rho, 9, 6, t1, t2, tg, form = 'rho')
    
#     # apply the second hadamard to the ancillas
#     current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
#     current_rho = rad_error(current_rho, t1, t2, tg)

#     # probability of the state measurments from the density matrix are defined as Tr(p*rho)
#     state_probs = np.array([])
#     tot = 0
#     for i in range(len(current_rho)):
#         tot += np.abs(current_rho[i, i])
#         state_probs = np.append(state_probs, current_rho[i,i])

#     # Measure the ancilla qubits and collapse them
#     collapsed_state = collapse_ancilla(np.sqrt(state_probs), 3)
#     if info:
#         print('Collapsed state after ancilla measurement:')
#         print_state_info(collapsed_state, 10)
#         print(' - ')

#     # Create our new density matrix after collapsing ancilla qubits
#     rho = np.kron(collapsed_state, collapsed_state[np.newaxis].conj().T)

#     # apply an error for time taken to collapse ancilla
#     rho = rad_error(rho, t1, t2, tg)

#     # How many total qubits are in our vector representation
#     n = int(np.log(len(collapsed_state))/np.log(2))

#     # Measure the three ancilla qubits
#     # Applying the Z gate operation on a specific qubit
#     bits = vector_state_to_bit_state(collapsed_state, n)[0][0]

#     # find index
#     m_one = 0
#     m_two = 0
#     m_three = 0
#     if bits[7] == '1':
#         m_one = 1
#     if bits[8] == '1':
#         m_two = 1
#     if bits[9] == '1':
#         m_three = 1

#     # Which qubit do we perform the Z gate on
#     index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) - 1

#     # if no error occurs we dont need to apply a correction
#     if index == -1:
#         final_rho = rho

#     else:
#         # apply the z gate depending on index
#         operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
#             np.identity(2**(n-3-index-1)), np.identity(2**3))))

#         final_rho = np.dot(operation, np.dot(rho, operation.conj().T))
#         final_rho = rad_error(final_rho, t1, t2, tg) # apply an error for correction gate time


#     # probability of the state measurments from the density matrix are defined as Tr(p*rho)
#     state_probs = np.array([])
#     tot = 0
#     for i in range(len(final_rho)):
#         tot += np.abs(final_rho[i,i])
#         state_probs = np.append(state_probs, final_rho[i,i])

#     final_state_z = np.sqrt(state_probs)
#     if info:
#         print('Final state after Z correction:')
#         print_state_info(final_state_z, 10)
#         print('- - -')

#     # Reset the ancilla qubits:
#     initial_state = ancilla_reset(final_state_z, 3)
#     if info:
#         print('Initial state before X correction:')
#         print_state_info(initial_state, 10)
#         print(' - ')

#     ### Implements the 7 Qubit Steane bit correction code using line connectivity
#     if info:
#         print('Applying Stabilizer operators 4-6.')
#         print('...')
    
#     # Create our new density matrix with reset ancillas
#     initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

#     # - - - - - - - - - - # X Error Correction # - - - - - - - - - - #

#     # apply the first hadamard to the ancillas
#     ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
#     current_rho = np.dot(ancilla_hadamard, np.dot(initial_rho, ancilla_hadamard.conj().T))
#     current_rho = rad_error(current_rho, t1, t2, tg)

#     # apply the control stabilizer gates to current_rho

#     # apply K4 first:
#     current_rho = line_rad_CZ(current_rho, 7, 3, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 7, 4, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 7, 5, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 7, 6, t1, t2, tg, form = 'rho')

#     # apply K5:
#     current_rho = line_rad_CZ(current_rho, 8, 0, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 8, 2, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 8, 4, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 8, 6, t1, t2, tg, form = 'rho')

#     # apply K6:
#     current_rho = line_rad_CZ(current_rho, 9, 1, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 9, 2, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 9, 5, t1, t2, tg, form = 'rho')
#     current_rho = line_rad_CZ(current_rho, 9, 6, t1, t2, tg, form = 'rho')

#     # apply the second hadamard to the ancillas
#     current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
#     current_rho = rad_error(current_rho, t1, t2, tg)

#     # probability of the state measurments from the density matrix are defined as Tr(p*rho)
#     state_probs = np.array([])
#     tot = 0
#     for i in range(len(current_rho)):
#         tot += np.abs(current_rho[i, i])
#         state_probs = np.append(state_probs, current_rho[i,i])

#     # Measure the ancilla qubits and collapse them
#     collapsed_state = collapse_ancilla(np.sqrt(state_probs), 3)
#     if info:
#         print('Collapsed state after ancilla measurement:')
#         print_state_info(collapsed_state, 10)
#         print(' - ')

#     # Create our new density matrix after collapsing ancilla qubits
#     rho = np.kron(collapsed_state, collapsed_state[np.newaxis].conj().T)

#     # apply an error for time taken to collapse ancilla
#     rho = rad_error(rho, t1, t2, tg)

#     # How many total qubits are in our vector representation
#     n = int(np.log(len(collapsed_state))/np.log(2))

#     # Measure the three ancilla qubits
#     # Applying the X gate operation on a specific qubit
#     bits = vector_state_to_bit_state(collapsed_state, n)[0][0]

#     # find index
#     m_four = 0
#     m_five = 0
#     m_six = 0
#     if bits[7] == '1':
#         m_four = 1
#     if bits[8] == '1':
#         m_five = 1
#     if bits[9] == '1':
#         m_six = 1

#     # Which qubit do we perform the Z gate on
#     index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) - 1

#     # if no error occurs we dont need to apply a correction
#     if index == -1:
#         final_rho = rho

#     else:
#         # apply the z gate depending on index
#         operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
#             np.identity(2**(n-3-index-1)), np.identity(2**3))))

#         final_rho = np.dot(operation, np.dot(rho, operation.conj().T))
#         final_rho = rad_error(final_rho, t1, t2, tg) # apply an error for correction gate time
        

#     # probability of the state measurments from the density matrix are defined as Tr(p*rho)
#     state_probs = np.array([])
#     tot = 0
#     for i in range(len(final_rho)):
#         tot += np.abs(final_rho[i,i])
#         state_probs = np.append(state_probs, final_rho[i,i])

#     final_state_x = np.sqrt(state_probs)
#     if info:
#         print('Final state after X correction:')
#         print_state_info(final_state_x, 10)

#     final_state = final_state_x
    
#     return final_rho, final_state
