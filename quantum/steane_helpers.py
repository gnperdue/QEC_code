# The functions in this file are useful when implementing the seven qubit steane code

import numpy as np
import random
from quantum.qec_helpers import *
from quantum.gates import *


# - - - - - - - - - -  Useful variables - - - - - - - - - - #

zero = np.array([[1, 0]])
one = np.array([[0, 1]])

# Setting up the 6 Stabilizer Operators for the 7-qubit Steane Code
k_one = np.kron(sigma_I, np.kron(sigma_I, np.kron(sigma_I, np.kron(
    sigma_x, np.kron(sigma_x, np.kron(sigma_x, sigma_x))))))
k_two = np.kron(sigma_x, np.kron(sigma_I, np.kron(sigma_x, np.kron(
    sigma_I, np.kron(sigma_x, np.kron(sigma_I, sigma_x))))))
k_three = np.kron(sigma_I, np.kron(sigma_x, np.kron(sigma_x, np.kron(
    sigma_I, np.kron(sigma_I, np.kron(sigma_x, sigma_x))))))
k_four = np.kron(sigma_I, np.kron(sigma_I, np.kron(sigma_I, np.kron(
    sigma_z, np.kron(sigma_z, np.kron(sigma_z, sigma_z))))))
k_five = np.kron(sigma_z, np.kron(sigma_I, np.kron(sigma_z, np.kron(
    sigma_I, np.kron(sigma_z, np.kron(sigma_I, sigma_z))))))
k_six = np.kron(sigma_I, np.kron(sigma_z, np.kron(sigma_z, np.kron(
    sigma_I, np.kron(sigma_I, np.kron(sigma_z, sigma_z))))))

### Gate operations for steane code using 3 ancillas ###

# phase correction gates
control_k_one = np.kron(np.identity(2**7), np.kron(np.kron(zero, zero.T), np.identity(2**2))) + np.kron(
    k_one, np.kron(np.kron(one, one.T), np.identity(2**2)))

control_k_two = np.kron(np.identity(2**7), np.kron(np.identity(2), np.kron(np.kron(
    zero, zero.T), np.identity(2)))) + np.kron(k_two, np.kron(np.identity(2), np.kron(
    np.kron(one, one.T), np.identity(2))))

control_k_three = np.kron(np.identity(2**7), np.kron(np.identity(2**2), np.kron(zero, zero.T))) + np.kron(
    k_three, np.kron(np.identity(2**2), np.kron(one, one.T)))

# bit correction gates
control_k_four = np.kron(np.identity(2**7), np.kron(np.kron(zero, zero.T), np.identity(2**2))) + np.kron(
    k_four, np.kron(np.kron(one, one.T), np.identity(2**2)))

control_k_five = np.kron(np.identity(2**7), np.kron(np.identity(2), np.kron(np.kron(
    zero, zero.T), np.identity(2)))) + np.kron(k_five, np.kron(np.identity(2), np.kron(
    np.kron(one, one.T), np.identity(2))))

control_k_six = np.kron(np.identity(2**7), np.kron(np.identity(2**2), np.kron(zero, zero.T))) + np.kron(
    k_six, np.kron(np.identity(2**2), np.kron(one, one.T)))

    
# - - - - - - - - - -  Initializations - - - - - - - - - - #

### Initializes the 10 qubit (7 physical, 3 ancilla) qubit system ###
def initialize_steane_logical_state(initial_state):
    # initial_state: initial state of your 7 qubits qubit that you want to use as your logical state combined with ancillas
    
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, ancilla_syndrome)

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system[0])

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(control_k_one, np.dot(control_k_two, np.dot(control_k_three, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)
    
    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    # Using this for superposition states, doesnt do anything for |0> initial states 
    # becuase they are already +1 eigenstates of Z
    if (initial_state != np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
        zero, np.kron(zero, np.kron(zero, zero))))))[0]).all():
            final_vector_state = steane_bit_correction(final_vector_state)
    
    # apply global phase correction:
    z_bar = np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(
        sigma_z, np.kron(sigma_z, sigma_z))))))
    z_bar = np.kron(z_bar, np.identity(2**3))
    final_vector_state = np.dot(z_bar, final_vector_state)
       
    return final_vector_state



# - - - - - - - - - -  Errors - - - - - - - - - - #

### Applies a random Z rotation to one of the physical qubits in your system (randomly) (works for both n= 10 and 13 qubits) ###
def phase_flip_error(logical_state, n):
    # logical_state: The logical state of the three qubit system you wish to apply the error to
    # n: The number of qubits in your logical system
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,6)
    
    if error_index == -1:
        # No error occurs in this case
        errored_logical_state = logical_state
        print('No phase flip error occured.')
    else:
        # Create the error as a gate operation
        error_gate = np.kron(np.identity(2**(error_index)), np.kron(sigma_z, np.identity(2**(n-error_index-1))))

        # Apply the error to the qubit (no error may occur)
        errored_logical_state = np.dot(error_gate, logical_state)
    
        print('Phase flip error on qubit: ', error_index)

    return errored_logical_state, error_index


### Applies a random X rotation to one of the physical qubits in your system (randomly) (works for both n= 10 and 13 qubits) ###
def bit_flip_error(logical_state, n):
    # logical_state: The logical state of the three qubit system you wish to apply the error to
    # n: The number of qubits in your logical system
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,6)
    
    if error_index == -1:
        # No error occurs in this case
        errored_logical_state = logical_state
        print('No bit flip error occured.')

    else:
        # Create the error as a gate operation
        error_gate = np.kron(np.identity(2**(error_index)), np.kron(sigma_x, np.identity(2**(n-error_index-1))))
        
        # Apply the error to the qubit (no error may occur)
        errored_logical_state = np.dot(error_gate, logical_state)
        print('Bit flip error on qubit: ', error_index)
        
    return errored_logical_state, error_index

# - - - - - - - - - - 3 ancilla error correction protocols - - - - - - - - - - #

### Corrects for a single phase flip error in the 7 qubit steane code with 3 ancillas ###
def steane_phase_correction(logical_state):
    # logical_state: The vector state representation of your full system

    full_system = logical_state
    
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(control_k_one, np.dot(control_k_two, np.dot(control_k_three, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)


    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)
    
    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)
    
    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    corrected_vector_state = remove_small_values(final_vector_state)
    
    return corrected_vector_state


### Corrects for a single bit flip error in the 7 qubit steane code with 3 ancillas ###
def steane_bit_correction(logical_state):
    # logical_state: The vector state representation of your full system

    full_system = logical_state
    
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(control_k_four, np.dot(control_k_five, np.dot(control_k_six, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)


    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)
    
   # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    # remove small values from state
    corrected_vector_state = remove_small_values(final_vector_state)
    
    return corrected_vector_state


# - - - - - - - - - - Larger 6 ancilla Steane code implementation - - - - - - - - - - #
##### - - - - - - NEEDS TO BE CORRECTED TO MATCH THE METHOD IN THE 10 QUBIT STEAN - - - - #####
### Gate operations for larger steane code using 6 ancillas ###
# (first 3 are controlled by  first 3 ancilla, other 3 are controlled by the other 3 ancilla)

# phase correction gates 
larger_control_k_one = np.kron(np.identity(2**7), np.kron(np.kron(zero, zero.T), np.identity(2**5))) + np.kron(
    k_one, np.kron(np.kron(one, one.T), np.identity(2**5)))
                        
larger_control_k_two = np.kron(np.identity(2**7), np.kron(np.identity(2), np.kron(np.kron(
    zero, zero.T), np.identity(2**4)))) + np.kron(k_two, np.kron(np.identity(2), np.kron(
    np.kron(one, one.T), np.identity(2**4))))
                        
larger_control_k_three = np.kron(np.identity(2**7), np.kron(np.identity(2**2), np.kron(np.kron(
    zero, zero.T), np.identity(2**3)))) + np.kron(k_three, np.kron(np.identity(2**2), np.kron(np.kron(
    one, one.T), np.identity(2**3))))

# bit correction gates
larger_control_k_four = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(
    np.kron(zero, zero.T), np.identity(2**2)))) + np.kron(
    k_four, np.kron(np.identity(2**3), np.kron(np.kron(one, one.T), np.identity(2**2))))

larger_control_k_five = np.kron(np.identity(2**7), np.kron(np.identity(2**4), np.kron(np.kron(
    zero, zero.T), np.identity(2)))) + np.kron(k_five, np.kron(np.identity(2**4), np.kron(
    np.kron(one, one.T), np.identity(2))))

larger_control_k_six = np.kron(np.identity(2**7), np.kron(np.identity(2**5), np.kron(zero, zero.T))) + np.kron(
    k_six, np.kron(np.identity(2**5), np.kron(one, one.T)))

### Splits the state up into vectors and takes only those that have '000000' as the ancilla measurement (using 6 ancilla) ###
def format_larger_state(logical_state):
    # logical_state: The logical state of the 13 qubit system
    
    n = 13 # Total number of qubits in the system

    # Take our vector and find the bit strings that represent it
    logical_bits, state_indices, logical_vector_state = vector_state_to_bit_state(logical_state, 13)
    
    # Finding the logical bits that contain '000000' in the end
    x=0
    for j in range(len(logical_bits)):
        if logical_bits[j][7:13] == '000000':
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

    # take out the vectors that do not have '000000' as the 6 ancilla bits
    for j in range(num_rows):
        if vector_state_to_bit_state(all_vector_states[j], 13)[0] not in final_bits : 
            all_vector_states[j].fill(0)

    # combine the vector states again
    final_vector_state = np.zeros((2**(n),), dtype=complex)
    for j in range(num_rows):
        final_vector_state = final_vector_state + all_vector_states[j]


    logical_bits = vector_state_to_bit_state(final_vector_state,7)[0]
    
    return final_vector_state


### Initializes the 13 qubit (7 physical, 6 ancilla) qubit system ###
def initialize_larger_steane_code(initial_state):
    # initial_state: initial state of your 7 qubits qubit that you want to use as your logical state combined with ancillas
    
    n = 13 # Total number of qubits in our system
    
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, np.kron(ancilla_syndrome, ancilla_syndrome))

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(
        np.kron(hadamard, np.kron(hadamard, hadamard)), np.identity(2**3)))

    full_system = np.dot(ancilla_hadamard, full_system[0])

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(larger_control_k_one, np.dot(larger_control_k_two, np.dot(larger_control_k_three, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 13)
    
    # Here we take the vector state and separate it into vectors so that we can apply a phase flip to designated qubits

    x = 0 # used to keep track of first indice where vector_state is non-zero

    for i in range(len(vector_state)):
        if vector_state[i] != 0: 
            # initialize the vector that will hold the single non-zero value in the proper spot
            value_position = np.zeros((1,2**n), dtype=complex) 
            value_position[:,i] = vector_state[i] # insert the non-zero value in the correct spot
            # Add the value position vector to an array of all the error places
            if x == 0:
                all_vector_states = value_position
            else:
                all_vector_states = np.append(all_vector_states, value_position , axis=0)
            x+=1

    # find the number of rows and columns in the all error state array so that we can loop over the rows later
    num_rows, num_cols = all_vector_states.shape
    
    # initialize the final vector state as all 0s so we can add in the values to designated spots
    final_vector_state = np.zeros((2**(n),), dtype=complex)

    # Measure the three ancilla qubits
    # Applying the Z gate operation on a qubit depending on the ancilla measuremnts
    for j in range(num_rows):
        # find index
        m_one = 0
        m_two = 0
        m_three = 0
        if bits[j][7] == '1':
            m_one = 1
        if bits[j][8] == '1':
            m_two = 1
        if bits[j][9] == '1':
            m_three = 1

        # Which qubit do we perform the Z gate on
        index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) -1

        # if no error occurs we dont need to apply a correction
        if index == -1:
            final_vector_state = final_vector_state + all_vector_states[j]
        else:
            # apply the z gate depending on index
            operation = np.kron(np.identity(2**index), np.kron(sigma_z, np.kron(
                np.identity(2**(n-6-index-1)), np.identity(2**6))))

            all_vector_states[j] = np.dot(operation, all_vector_states[j])

            # combine the vector states again
            final_vector_state = final_vector_state + all_vector_states[j]


    logical_bits, state_indices, logical_vector_state = vector_state_to_bit_state(final_vector_state, 13)
    
    final_vector_state = format_larger_state(logical_vector_state)

    return final_vector_state


### Applies the simultaneous error correction code ###
def simultaneous_error_correction(logical_state):
    # logical_state: the full logical state of the 13 qubit system after initialization

    n = 13 # Total number of qubits in our system

    full_system = logical_state
    
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(
        np.kron(hadamard, np.kron(hadamard, hadamard)), np.kron(hadamard, np.kron(hadamard, hadamard))))

    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(larger_control_k_one, np.dot(larger_control_k_two, np.dot(
        larger_control_k_three, np.dot(larger_control_k_four, np.dot(larger_control_k_five, np.dot(larger_control_k_six, full_system))))))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)


    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 13)

    # Here we take the vector state and separate it into vectors so that we can apply a phase flip to designated qubits

    # - The part below currently takes about 1 min to 1 min 30 sec - #

    x = 0 # used to keep track of first indice where vector_state is non-zero

    for i in range(len(vector_state)):
        if vector_state[i] != 0: 
            # initialize the vector that will hold the single non-zero value in the proper spot
            value_position = np.zeros((1,2**n), dtype=complex) 
            value_position[:,i] = vector_state[i] # insert the non-zero value in the correct spot
            # Add the value position vector to an array of all the error places
            if x == 0:
                all_vector_states = value_position
            else:
                all_vector_states = np.append(all_vector_states, value_position , axis=0)
            x+=1

    # find the number of rows and columns in the all error state array so that we can loop over the rows later
    num_rows, num_cols = all_vector_states.shape

    # initialize the final vector state as all 0s so we can add in the values to designated spots
    final_vector_state = np.zeros((2**(n),), dtype=complex)

    # Measure the three ancilla qubits
    # Applying the X gate operation on a qubit depending on the ancilla measuremnts
    for j in range(num_rows):
        # find index
        m_one = 0
        m_two = 0
        m_three = 0
        m_four = 0
        m_five = 0
        m_six = 0
        if bits[j][7] == '1':
            m_one = 1
        if bits[j][8] == '1':
            m_two = 1
        if bits[j][9] == '1':
            m_three = 1
        if bits[j][10] == '1':
            m_four = 1
        if bits[j][11] == '1':
            m_five = 1
        if bits[j][12] == '1':
            m_six = 1

        # Which qubit do we perform the Z gate on
        phase_index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) -1

        # Which qubit do we perform the X gate on
        bit_index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) -1

        # if no error occurs we dont need to apply a correction
        if (phase_index == -1) and (bit_index == -1):
            final_vector_state = final_vector_state + all_vector_states[j][:]

        # Phase error occurs but no bit error
        elif (phase_index != -1) and (bit_index == -1):
            # apply the z gate depending on index
            operation = np.kron(np.identity(2**(phase_index)), np.kron(sigma_z, np.kron(
                np.identity(2**(n-6-phase_index-1)), np.identity(2**6))))

            all_vector_states[j][:] = np.dot(operation, all_vector_states[j][:])

            # combine the vector states again
            final_vector_state = final_vector_state + all_vector_states[j][:]

        # Bit error occurs but no phase error
        elif (phase_index == -1) and (bit_index != -1):
            # apply the x gate depending on bit_index
            operation = np.kron(np.identity(2**(bit_index)), np.kron(sigma_x, np.kron(
                np.identity(2**(n-6-bit_index-1)), np.identity(2**6))))

            all_vector_states[j][:] = np.dot(operation, all_vector_states[j][:])

            # combine the vector states again
            final_vector_state = final_vector_state + all_vector_states[j][:]

        # Both phase and bit errors occur
        else:
            # apply the z gate depending on phase_index
            phase_operation = np.kron(np.identity(2**(phase_index)), np.kron(sigma_z, np.kron(
                np.identity(2**(n-6-phase_index-1)), np.identity(2**6))))

            # apply the x gate depending on bit_index
            bit_operation = np.kron(np.identity(2**(bit_index)), np.kron(sigma_x, np.kron(
                np.identity(2**(n-6-bit_index-1)), np.identity(2**6))))

            all_vector_states[j][:] = np.dot(phase_operation, np.dot(bit_operation, all_vector_states[j][:]))

            # combine the vector states again
            final_vector_state = final_vector_state + all_vector_states[j][:]


    logical_bits, state_indices, logical_vector_state = vector_state_to_bit_state(final_vector_state, 13)

    # Used to remove the smaller values after error correction
    x=0
    for j in range(len(logical_vector_state)):
        if (abs(logical_vector_state[j]) > 1e-15): 
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

    # combine the vector states again
    corrected_vector_state = np.zeros((2**(n),), dtype=complex)

    for j in range(num_rows):
        corrected_vector_state = corrected_vector_state + all_vector_states[j][:]

    return corrected_vector_state


# - - - - - - - - - - Steane Code Using Line Connectivity - - - - - - - - - - #

# Define the Stabilizer Operators as CNOT gates between line adjacent qubits 
# (remember that the non-adj CNOT calculation is using line connectivity)
K1_line_operation = np.dot(flipped_non_adj_CNOT(7, 3, 10), np.dot(flipped_non_adj_CNOT(7, 4, 10), np.dot(
    flipped_non_adj_CNOT(7, 5, 10), flipped_adj_CNOT(7, 6, 10))))
K2_line_operation = np.dot(flipped_non_adj_CNOT(8, 0, 10), np.dot(flipped_non_adj_CNOT(8, 2, 10), np.dot(
    flipped_non_adj_CNOT(8, 4, 10), flipped_non_adj_CNOT(8, 6, 10))))
K3_line_operation = np.dot(flipped_non_adj_CNOT(9, 1, 10), np.dot(flipped_non_adj_CNOT(9, 2, 10), np.dot(
    flipped_non_adj_CNOT(9, 5, 10), flipped_non_adj_CNOT(9, 6, 10))))

K4_line_operation = np.dot(non_adj_CZ(7, 3, 10), np.dot(non_adj_CZ(7, 4, 10), np.dot(
    non_adj_CZ(7, 5, 10), adj_CZ(7, 6, 10))))
K5_line_operation =np.dot(non_adj_CZ(8, 0, 10), np.dot(non_adj_CZ(8, 2, 10), np.dot(
    non_adj_CZ(8, 4, 10), non_adj_CZ(8, 6, 10))))
K6_line_operation =np.dot(non_adj_CZ(9, 1, 10), np.dot(non_adj_CZ(9, 2, 10), np.dot(
    non_adj_CZ(9, 5, 10), non_adj_CZ(9, 6, 10))))


### Initializes the 10 qubit (7 physical, 3 ancilla) qubit system ###
def initialize_steane_line_conn(initial_state):
    # initial_state: initial state of your 7 qubits qubit that you want to use as your logical state combined with ancillas
    
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, ancilla_syndrome)

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system[0])

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(K1_line_operation, np.dot(K2_line_operation, np.dot(K3_line_operation, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)
    
    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    # Using this for superposition states, doesnt do anything for |0> initial states 
    # becuase they are already +1 eigenstates of Z
    if (initial_state != np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
        zero, np.kron(zero, np.kron(zero, zero))))))[0]).all():
            final_vector_state = steane_bit_correction(final_vector_state)
    
    # apply global phase correction:
    z_bar = np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(
        sigma_z, np.kron(sigma_z, sigma_z))))))
    z_bar = np.kron(z_bar, np.identity(2**3))
    final_vector_state = np.dot(z_bar, final_vector_state)
       
    return final_vector_state


### Implements the 7 Qubit Steane phase correction code using line connectivity
def steane_line_conn_phase_correction(logical_state):
    # logical_state: The vector state representation of your 10 qubit system 
                    # (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    
    full_system = logical_state
    

    # - - - - - - - - - - # Z Error Correction # - - - - - - - - - - #
    
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(K1_line_operation, np.dot(K2_line_operation, np.dot(K3_line_operation, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)


    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)
    
    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    corrected_vector_state = remove_small_values(final_vector_state)
    
    return corrected_vector_state



### Implements the 7 Qubit Steane bit correction code using line connectivity
def steane_line_conn_bit_correction(logical_state):
    # logical_state: The vector state representation of your 10 qubit system 
                    # (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    
    full_system = logical_state
    
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))

    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(K4_line_operation, np.dot(K5_line_operation, np.dot(K6_line_operation, full_system)))
    # full_system = np.dot(control_k_four, np.dot(control_k_five, np.dot(control_k_six, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)
    
    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)
    
   # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    # remove small values from state
    corrected_vector_state = remove_small_values(final_vector_state)
    
    return corrected_vector_state

# - - - - - - - - - - Steane Code Using Grid Connectivity - - - - - - - - - - #
    
# Define the Stabilizer Operators as CNOT gates between grid adjacent qubits
# (remember that the non-adj CNOT calculation is using line connectivity)

# SWAPS needed for K1
swap_a0_6 = np.dot(CNOT(7, 6, 10), np.dot(CNOT(6, 7, 10), CNOT(7, 6, 10)))
swap_6_a1 = np.dot(CNOT(8, 6, 10), np.dot(CNOT(6, 8, 10), CNOT(8, 6, 10)))
# Create K1 operator
K1_grid_operation = np.dot(CNOT(7, 3, 10), np.dot(np.dot(
    np.dot(swap_a0_6, swap_6_a1), np.dot(CNOT(8, 4, 10), np.dot(swap_6_a1, swap_a0_6))), np.dot(
    CNOT(7, 5, 10), CNOT(7, 6, 10))))

# SWAPS needed for K2
swap_a1_4 = np.dot(CNOT(8, 4, 10), np.dot(CNOT(4, 8, 10), CNOT(8, 4, 10)))
# Create K2 operator
K2_grid_operation = np.dot(CNOT(8, 0, 10), np.dot(
    np.dot(swap_a1_4, np.dot(CNOT(4, 2, 10), swap_a1_4)), np.dot(
    CNOT(8, 4, 10), CNOT(8, 6, 10))))

# SWAPS needed for K3
swap_a2_4 = np.dot(CNOT(9, 4, 10), np.dot(CNOT(4, 9, 10), CNOT(9, 4, 10)))
# Create K3 operator
K3_grid_operation = np.dot(CNOT(9, 1, 10), np.dot(
    np.dot(swap_a2_4, np.dot(CNOT(4, 2, 10), swap_a2_4)), np.dot(
    CNOT(9, 5, 10), CNOT(9, 6, 10))))



# SWAPS needed for K4 are the same as K1
# Create K1 operator
K4_grid_operation = np.dot(non_adj_CZ(7, 3, 10), np.dot(np.dot(
    np.dot(swap_a0_6, swap_6_a1), np.dot(non_adj_CZ(8, 4, 10), np.dot(swap_6_a1, swap_a0_6))), np.dot(
    non_adj_CZ(7, 5, 10), adj_CZ(7, 6, 10))))

# SWAPS needed for K5 are the same as K2
# Create K2 operator
K5_grid_operation = np.dot(non_adj_CZ(8, 0, 10), np.dot(
    np.dot(swap_a1_4, np.dot(non_adj_CZ(4, 2, 10), swap_a1_4)), np.dot(
    non_adj_CZ(8, 4, 10), non_adj_CZ(8, 6, 10))))

# SWAPS needed for K6 are the same as K3
# Create K3 operator
K6_grid_operation = np.dot(non_adj_CZ(9, 1, 10), np.dot(
    np.dot(swap_a2_4, np.dot(non_adj_CZ(4, 2, 10), swap_a2_4)), np.dot(
    non_adj_CZ(9, 5, 10), non_adj_CZ(9, 6, 10))))


### Initializes the 10 qubit (7 physical, 3 ancilla) qubit system ###
def initialize_steane_grid_conn(initial_state):
    # initial_state: initial state of your 7 qubits qubit that you want to use as your logical state combined with ancillas
    
    
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, ancilla_syndrome)

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system[0])

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(K1_grid_operation, np.dot(K2_grid_operation, np.dot(K3_grid_operation, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)
    
  # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    # Using this for superposition states, doesnt do anything for |0> initial states 
    # becuase they are already +1 eigenstates of Z
    if (initial_state != np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
        zero, np.kron(zero, np.kron(zero, zero))))))[0]).all():
            final_vector_state = steane_bit_correction(final_vector_state)
    
    # apply global phase correction:
    z_bar = np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(
        sigma_z, np.kron(sigma_z, sigma_z))))))
    z_bar = np.kron(z_bar, np.identity(2**3))
    final_vector_state = np.dot(z_bar, final_vector_state)
       
    return final_vector_state

### Implements the 7 Qubit Steane phase correction code using grid connectivity
def steane_grid_conn_phase_correction(logical_state):
    # logical_state: The vector state representation of your 10 qubit system 
                    # (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    
    full_system = logical_state
    

    # - - - - - - - - - - # Z Error Correction # - - - - - - - - - - #
    
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(K1_grid_operation, np.dot(K2_grid_operation, np.dot(K3_grid_operation, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)


     # remove small values after applying operations
    vector_state = remove_small_values(vector_state)
    
    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    corrected_vector_state = remove_small_values(final_vector_state)
    
    return corrected_vector_state


### Implements the 7 Qubit Steane bit correction code using grid connectivity
def steane_grid_conn_bit_correction(logical_state):
    # logical_state: The vector state representation of your 10 qubit system 
                    # (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    
    full_system = logical_state
    
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))

    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(K4_grid_operation, np.dot(K5_grid_operation, np.dot(K6_grid_operation, full_system)))
    # full_system = np.dot(control_k_four, np.dot(control_k_five, np.dot(control_k_six, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)
    
    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)
    
   # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))
    
    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)
    
    # remove small values from state
    corrected_vector_state = remove_small_values(final_vector_state)
    
    return corrected_vector_state

