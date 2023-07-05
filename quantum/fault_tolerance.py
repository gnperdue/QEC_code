# This file will contain functions that are useful when implementing fault tolerance quantum circuits

import numpy as np
import random
from quantum.qec_helpers import *
from quantum.gates import *
from quantum.errors import *

### - - - - - - - - - - Fault Tolerant 7 qubit Steane code - - - - - - - - - - ###

# Initializes and encodes the ancilla block (5 qubits)
def initialize_ancilla_block():
    # Choose a spot that the single bit flip error occurs in the circuit
    spot_x = random.randint(0, 1)
    spot_z = random.randint(0, 1)

    # initialize ancilla block to |00000>
    ancilla_block = np.kron(zero, np.kron(zero, np.kron(zero, np.kron(zero, zero))))
    
    # Error spot location 0 for X
    if spot_x == 0:
        ancilla_block = random_qubit_x_error(ancilla_block)[0]
       
    # Error spot location 0 for Z
    if spot_z == 0:
        ancilla_block = random_qubit_z_error(ancilla_block)[0]

    # create the hadamard gate
    h = np.kron(np.identity(2**3), np.kron(hadamard, np.identity(2)))
    # apply hadamard
    ancilla_block = np.dot(h, ancilla_block)
    
    # Error spot location 1 for X
    if spot_x == 1:
        ancilla_block = random_qubit_x_error(ancilla_block)[0]
    
    # Error spot location 1 for Z
#     if spot_z == 1:
#         ancilla_block = random_qubit_z_error(ancilla_block)[0]
        
    # create the CNOT gates
    cnot_gates = np.dot(CNOT(0, 4, 5), np.dot(CNOT(1, 0 ,5), np.dot(CNOT(2, 1, 5), np.dot(CNOT(3, 4, 5), CNOT(3, 2, 5)))))
    # apply  CNOT gates
    ancilla_block = np.dot(cnot_gates, ancilla_block)
    
    return ancilla_block


# Loops over the initialization of the ancilla block until M measures 0 (12 qubits)
def ancilla_loop(logical_state):
    # logical_state: The vector state representation of you 12 qubit system
    a = '1'
    while a == '1':
        state = initialize_ancilla_block(logical_state)
        a = vector_state_to_bit_state(state, 12)[0][0][11:]
    return state


### Splits the state up into vectors and takes only those that have '0' as the ancilla measurement (using n-7 ancilla) ###
def format_state(logical_state, n):
    # logical_state: The logical state of the 10 qubit system
    #n: Total number of qubits in the system

    # Take our vector and find the bit strings that represent it
    logical_bits, state_indices, logical_vector_state = vector_state_to_bit_state(logical_state, n)
    
    # Finding the logical bits that contain '000' in the end
    x=0
    for j in range(len(logical_bits)):
        if logical_bits[j][7:n] == ('0' * (n-7)):
            if x == 0:
                final_bits = logical_bits[j]
            else:
                final_bits = np.append(final_bits, logical_bits[j])
            x+=1
    
    # Take the vector and split it into individual vectors that contain only a single non zero value in the same spot
    x=0
    for j in range(len(logical_vector_state)):
        if logical_vector_state[j] != 0: 
            # initialize the vector that will hold the single non-zero value in the proper spot
            value_position = np.zeros((1,2**n), dtype=complex) 
            value_position[:,j] = logical_vector_state[j] # insert the non-zero value in the correct spot
            # Add the value position vector to an array of all the error places
            if x == 0:
                all_vector_states = value_position
            else:
                all_vector_states = np.append(all_vector_states, value_position , axis=0)
            x+=1

    # find the number of rows and columns in the all error state array so that we can loop over the rows later
    num_rows, num_cols = all_vector_states.shape

    # take out the vectors that do not have '0' as the ancilla bits
    for j in range(num_rows):
        if vector_state_to_bit_state(all_vector_states[j][:], n)[0] not in final_bits : 
            all_vector_states[j][:].fill(0)

    # combine the vector states again
    final_vector_state = np.zeros((2**(n),), dtype=complex)
    for j in range(num_rows):
        final_vector_state = final_vector_state + all_vector_states[j][:]
    
    return final_vector_state

