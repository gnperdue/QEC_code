import numpy as np
import random

### Initializes the three qubit logical state using an initial single qubit psi ###
def three_qubit_initialize_logical_state(initial_psi):
    # initial_psi: initial state of your single qubit that you want to use as your logical state (2 x 1)
    
    initial_ancilla_state = np.array([1,0]) # initializing the |0> state of the qubits
    
    # Initialize the 3 qubit logical state by using thr kronecker product
    initial_logical_state = np.kron(initial_psi, np.kron(initial_ancilla_state, initial_ancilla_state))

    # Setting up the 2 CNOT gates to initialize the correct logical qubit
    cnot_psi_qzero = np.kron(cnot, np.identity(2))
    cnot_qzero_qone = np.kron(np.identity(2), cnot)
    
    # Apply the CNOT gates to the kronecker product of the current 3 qubit state
    final_logical_state = np.dot(cnot_qzero_qone, np.dot(cnot_psi_qzero, initial_logical_state))
    
    return final_logical_state


### - - - - - - Usefull gate operations - - - - - - ###

# Pauli operators
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,1j],[-1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_I = np.identity(2)

# Hadamard Gate
hadamard = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]])

# CNOT gate
cnot = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])

### Implement a non-adjacent CNOT gate between 2 qubits in a system ###
def non_adj_CNOT(control, target, tot_qubits):
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # tot_qubits: total number of qubits in the system (if there are 5 qubits then put 5 ...)
    
    p = target - control # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end
    
    # Indexing over the values of p to get the first half of the formula
    for j in range(p):
        # Sets the next component of the matrix multiplication up
        next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
        
        # Adds the components to the array and multiplies them together
        if j == 0:
            all_dots = np.array([next_dot])
            gate = all_dots[j]
        else:
            all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = np.dot(gate, all_dots[j])
            
    # Indexing over values of p such that we get the 2nd half of the equation together
    for j in range(p - 2):
        gate = np.dot(gate, all_dots[p-j-2])
    
    # Squares the final matrix
    final_gate = np.dot(gate, gate)
    
    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    final_total_gate = np.kron(np.identity(2**(n1)), np.kron(final_gate, np.identity(2**(n2))))
    
    return final_total_gate


### Changes the vector state representation to the bit representation (only works for max of 5 right now) ###
def vector_state_to_bit_state(logical_state, k):
    # logical_state: the full logical state of the qubit system you wish to reduce (32 x 1)
    # k: the number of qubits you wish to reduce the system to (must be less than the full system size)

    # used to keep an index of where the non-zero element is in the vector representation
    index_of_element = np.array([]) 
    for i in range(logical_state.size):
        if logical_state[i] != 0:
            index_of_element = np.append(index_of_element, i)

    # How many total qubits are in our vector representation
    n = int(np.log(len(logical_state))/np.log(2))

    # Keeps track of the logical bits needed 
    # (i.e. a|000> + b|111> : 000 and 111 are considered separate and we will combine them)
    log_bit = np.array([])

    # Create the bits and make sure they have the correct number of '0's in front 
    for j in range(index_of_element.size):
        bits = bin(index_of_element[j].astype(int))
        bits = bits[2:]  # Remove the '0b' prefix

        if len(bits) < n:
            zeros = '0' * (n - len(bits))
            new_bits = zeros + bits[0:(k - (n - len(bits)))]
            new_bits = new_bits[0:k]
            log_bit = np.append(log_bit, new_bits)
        else:
            log_bit = np.append(log_bit, bits[0:k])

    return log_bit, index_of_element, logical_state


### - - - - - - Errors - - - - - - ###

### Applies a random X rotation to one of the three physical qubits in your system (randomly) ###
def three_qubit_random_qubit_x_error(logical_state):
    # logical_state: The logical state of the three qubit system you wish to apply the error to (8 x 1)
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,2)
    # Apply the error to the qubit (no error may occur)
    if error_index == 0:
        errored_logical_state = np.dot(np.kron(sigma_x, np.identity(4)), logical_state)
    elif error_index == 1:
        errored_logical_state = np.dot(np.kron(np.kron(np.identity(2), sigma_x), np.identity(2)), logical_state)
    elif error_index == 2:
        errored_logical_state = np.dot(np.kron(np.identity(4), sigma_x), logical_state)
    else:
        errored_logical_state = logical_state
        
    return errored_logical_state, error_index


### Applies an X rotation to one of the three physical qubits in your system (your choice for which qubit is errored) ###
def three_qubit_defined_qubit_x_error(logical_state, error_index):
    # logical_state: The logical state of the three qubit system you wish to apply the error to (8 x 1)
    # error_index: Which qubit you want the error to occur.
    
    # Apply the error to the qubit (no error may occur)
    if error_index == 0:
        errored_logical_state = np.dot(np.kron(sigma_x, np.identity(4)), logical_state)
    elif error_index == 1:
        errored_logical_state = np.dot(np.kron(np.kron(np.identity(2), sigma_x), np.identity(2)), logical_state)
    elif error_index == 2:
        errored_logical_state = np.dot(np.kron(np.identity(4), sigma_x), logical_state)
    else:
        errored_logical_state = logical_state
        
    return errored_logical_state, error_index


### Applies an arbitrary X rotation to all of the three physical qubits in your system ###
def three_qubit_coherent_x_rotation_error(logical_state, epsilon):
    # logical_state: state of the logical qubit before error occurs
    # epsilon: error constant in a coherent rotation error
    
    U = np.cos(epsilon) * sigma_I + 1j*np.sin(epsilon) * sigma_x # Create the Unitary error operator 
    E = np.kron(U, np.kron(U, U)) # Create the Error operator that will act on our logical qubit
    
    # Apply error
    errored_state = np.dot(E, logical_state)
    
    return errored_state, E, U


### - - - - - - Error Detection - - - - - - ###

### Applying the ancilla qubits to the three qubit system ###
def three_qubit_apply_ancillas(logical_state):
    # logical_state: the vector state representation of our 3 qubit system (8 x 1)
    
    # Extend our system to add in the 2 syndrome ancilla qubits
    full_system = np.kron(logical_state, np.kron(np.array([1,0]), np.array([1,0]))) 

    # Apply the CNOT gates needed to change the state of the syndrome ancilla 
    final_logical_state = np.dot(non_adj_CNOT(2,4,5), np.dot(non_adj_CNOT(0,4,5), 
                                                     np.dot(non_adj_CNOT(1,3,5), np.dot(non_adj_CNOT(0,3,5), full_system))))
    
    return final_logical_state


### Detects where the x rotation error occured from the vector from of the 5 qubit system ###
def three_qubit_detect_error_location_from_vector(logical_state):
    # logical_state: the logical state of our 3 qubit system with 2 ancillas (32 x 1)
    
    # Initialize error index
    error_index = -1

    if (logical_state[28] != 0) or (logical_state[0] != 0): # No error occured
        error_index = -1
        return error_index, print("No bit flip error occured.")
    elif (logical_state[15] != 0) or (logical_state[19] != 0): # Error on qubit 0
        error_index = 0
    elif (logical_state[22] != 0) or (logical_state[10] != 0): # Error on qubit 1
        error_index = 1
    elif (logical_state[25] != 0) or (logical_state[5] != 0): # Error on qubit 2
        error_index = 2
        
    return error_index, print("Bit flip error occured on qubit", error_index )


### Detects where the x rotation error occured from the bit form of the 5 qubit system ###
def three_qubit_detect_error_location_from_bit_state(logical_bits):
    # logical_bits: set of 5 qubits to detect errors with 2 ancilla within the 5 (00000)
    
    # Initialize error index
    error_index = -1
    
    if ((logical_bits[3] == '1') and (logical_bits[4] == '1')): # Error on qubit 0
        error_index = 0
    elif ((logical_bits[3] == '1') and (logical_bits[4] == '0')): # Error on qubit 1
        error_index = 1
    elif ((logical_bits[3] == '0') and (logical_bits[4] == '1')): # Error on qubit 2
        error_index = 2
    elif(logical_bits[3] and logical_bits[4] == '0'): # No error occured
        return error_index, print("No bit flip error occured.")
    
    return error_index, print("Bit flip error occured on qubit", error_index )


### - - - - - - Error Correction - - - - - - ###

### Correct for errors by applying full X rotation gate to the qubit where the error occured. ###
def three_qubit_correct_full_x_error(logical_state):
    # logical_state: the logical state of our 3 qubit system with 2 ancillas (32 x 1)

    # Find where the error occured using the error detection function
    qubit_index = three_qubit_detect_error_location_from_vector(logical_state)[0]
    
    if qubit_index == 0: # Error on qubit 0
        corrected_state = np.dot(np.kron(sigma_x, np.identity(16)), logical_state)
    elif qubit_index == 1: # Error on qubit 1
        corrected_state = np.dot(np.kron(np.identity(2), np.kron(sigma_x, np.identity(8))), logical_state)
    elif qubit_index == 2: # Error on qubit 2
        corrected_state = np.dot(np.kron(np.identity(4), np.kron(sigma_x, np.identity(4))), logical_state)
    else: # No error occured
        corrected_state = logical_state
    
    return corrected_state

### - - - - - - Outputting Information - - - - - - ###

### Outputting the information for the three qubit bit flip correcting code
def three_qubit_info(vector_error_state, vector_corrected_state):
    # vector_error_state: the logical state of our errored 3 qubit system with 2 ancillas (32 x 1)
    # vector_corrected_state: the logical state of our corrected 3 qubit system with 2 ancillas (32 x 1)
    
    # Find the 5 bit representation information for the errored vector state
    error_logical_bits, error_index, error_state = vector_state_to_bit_state(vector_error_state, 5)
    
    # Ouput the information for the 5 bit errored state
    if len(error_index) < 2:
        print('Full system error bit state:     ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0])
    else:
        print('Full system error bit state:     ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0], ' + ', 
              error_state[error_index[1].astype(int)], error_logical_bits[1])

    # Find the 5 bit representation information for the errored vector state
    corrected_logical_bits, corrected_index, corrected_state = vector_state_to_bit_state(vector_corrected_state, 5)
    
    # Ouput the information for the 5 bit corrected state
    if len(corrected_index) < 2:
        print('Full system corrected bit state: ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0])
    else:
        print('Full system corrected bit state: ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0], ' + ', 
              corrected_state[corrected_index[1].astype(int)], corrected_logical_bits[1])

    # Find the 3 bit representation information for the errored vector state
    error_logical_bits, error_index, error_state = vector_state_to_bit_state(vector_error_state, 3)
    
    # Ouput the information for the 3 bit errored state
    if len(error_index) < 2:
        print('Error bit state:                 ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0])
    else:
        print('Error bit state:                 ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0], '   + ', 
              error_state[error_index[1].astype(int)], error_logical_bits[1])
    
    # Find the 3 bit representation information for the corrected vector state
    corrected_logical_bits, corrected_index, corrected_state = vector_state_to_bit_state(vector_corrected_state, 3)
    
    # Ouput the information for the 3 bit corrected state
    if len(corrected_index) < 2:
        print('Corrected bit state:             ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0])
    else:
        print('Corrected bit state:             ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0], '   + ', 
              corrected_state[corrected_index[1].astype(int)], corrected_logical_bits[1])







