"""
QEC Helpers
"""
import random
import numpy as np
from general_qec.gates import sigma_x

# A few helpful states to use in initializing a single qubit (they have an exra
# dimension since some cases it is needed, but this can be removed fairly
# easily)
zero = np.array([1, 0])
one = np.array([0, 1])
superpos = 1/np.sqrt(2) * np.array([1, 1])


def vector_state_to_bit_state(logical_state, k):
    """
    Changes the vector state representation to the bit representation

    * logical_state: the full logical state of the n qubit system you wish to
    reduce (2^n x 1) - e.g. np.kron(one, one)
    * k: the number of qubits you wish to reduce the system to (must be less
    than the full system size) - e.g. 2 for np.kron(one, one)

    TODO: `k` is a confusing argument here...
    """
    # used to keep an index of where the non-zero element is in the vector
    # representation
    index_of_element = np.array([])
    for i in range(logical_state.size):
        if logical_state[i] != 0:
            index_of_element = np.append(index_of_element, i)

    # How many total qubits are in our vector representation
    n = int(np.log(len(logical_state))/np.log(2))

    # Keeps track of the logical bits needed
    # (i.e. a|000> + b|111> : 000 and 111 are considered separate and we will
    # combine them)
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



def print_state_info(logical_state, k):
    """
    Prints out the full state information with the amplitude of each state, and
    a total probability

    * logical_state: the full logical state of the n qubit system you wish to
    reduce (2^n x 1) - e.g. np.kron(one, one)
    * k: the number of qubits you wish to reduce the system to (must be less
    than the full system size) - e.g. 2 for np.kron(one, one)

    TODO: `k` is a confusing argument here...
    """
    bit_states, index, vector_state = \
        vector_state_to_bit_state(logical_state, k)
    non_zero_vector_state = vector_state[vector_state != 0]

    for j in range(len(bit_states)):
        print(bit_states[j], ': ', non_zero_vector_state[j])


def collapse_ancilla(logical_state, k):
    """
    Collapse the ancilla qubits to one of their states and return the
    vector representation (probability taken in to account

    * logical_state: The vector state representation of your full qubit
    system
    * k: number of ancillas in your system (at the end of the bit
    representation)
    """
    # How many total qubits are in our vector representation
    n = int(np.log(len(logical_state))/np.log(2))

    # Find all of the bit combinations that are in our vector state
    # representation
    all_bits, indices, logical_state = \
        vector_state_to_bit_state(logical_state, n)

    # create two empty arrays to store information
    organized_bits = np.array([])
    all_organized_bits = np.array([[]])

    # loop over our bit representations and organize them based on
    # whether or not they have the same ancilla qubits
    for j in range(int(len(all_bits))):
        organized_bits = all_bits
        for i in range(len(all_bits)):
            if all_bits[j][n-k:] != all_bits[i][n-k:]:
                organized_bits = np.delete(
                    organized_bits, organized_bits == all_bits[i])
        if j == 0:
            all_organized_bits = [organized_bits]
        else:
            all_organized_bits = np.append(
                all_organized_bits, [organized_bits], axis = 0)

    all_organized_bits = np.unique(all_organized_bits, axis=0)

    # finding our probability for measurement
    rows, cols = np.shape(all_organized_bits)
    probs = np.array([])
    for k in range(rows):
        summation = 0
        for j in range(cols):
            # TODO: `int(indices[all_bits == all_organized_bits[k][j]])`
            # --> generates a warning:
            # "DeprecationWarning: Conversion of an array with ndim > 0 to a
            # scalar is deprecated, and will error in future. Ensure you
            # extract a single element from your array before performing this
            # operation. (Deprecated NumPy 1.25.)" -> Think we fix this with:
            # `int(indices[all_bits == all_organized_bits[k][j]][0])`, but I
            # want to write a few more tests before changing.
            summation += np.abs(
                logical_state[
                    int(indices[all_bits == all_organized_bits[k][j]][0])
                ]
            )**2
        probs = np.append(probs, summation)

    # find which ancilla we will measure
    index = random.choices(all_organized_bits, weights=probs, k=1)
    index = np.where(all_organized_bits == index)[0][0]
    # set our collapsed state to that ancilla measurement
    collapsed_bits = all_organized_bits[index]

    # Here we take the vector state and separate it into vectors so that we
    # can manipulate it
    x = 0 # used to track first index where vector_state is non-zero

    for i in range(len(logical_state)):
        if logical_state[i] != 0:
            # initialize the vector that will hold the single non-zero value
            # in the proper spot
            value_position = np.zeros((2**n,), dtype=complex)
            # insert the non-zero value in the correct spot
            value_position[i,] = logical_state[i]
            # Add the value position vector to an array of all the error places
            if x == 0:
                all_vector_states = [value_position]
            else:
                all_vector_states = np.append(
                    all_vector_states, [value_position] , axis=0)
            x+=1

    # find the number of rows and columns in the all error state array so that
    # we can loop over the rows later
    num_rows, num_cols = np.array(all_vector_states).shape

    # take out the vectors that do not match our collapsed bit state
    for j in range(num_rows):
        if vector_state_to_bit_state(all_vector_states[j], n)[0] \
            not in collapsed_bits:
            all_vector_states[j][:].fill(0)

    # combine the vector states again
    collapsed_vector_state = np.zeros((2**(n),), dtype=complex)
    for j in range(num_rows):
        collapsed_vector_state = collapsed_vector_state + \
            all_vector_states[j][:]

    # normalizing our state
    pop = 0
    for i in range(len(probs)):
        pop += probs[i]

    norm = np.linalg.norm(collapsed_vector_state)
#     print_state_info(collapsed_vector_state, n)
#     print('pop: ', pop, 'norm: ', norm)
    collapsed_vector_state =  np.sqrt(pop) * (collapsed_vector_state/norm)

    return collapsed_vector_state


def ancilla_reset(logical_state, k):
    """
    Reset the ancilla qubits to '0'

    * logical_state: The vector state representation of your full qubit system
    * k: number of ancillas in your system (at the end of the bit
    representation)
    """
    # How many total qubits are in our vector representation
    n = int(np.log(len(logical_state))/np.log(2))

    reset_state = logical_state

    all_ancilla_bits = vector_state_to_bit_state(reset_state, n)[0]

    for j in range(len(all_ancilla_bits)):
        ancilla_bits = vector_state_to_bit_state(reset_state, n)[0][j]
        for i in range(n):
            if i >= n-k:
                if ancilla_bits[i] == '1':
                    reset_gate = np.kron(
                        np.identity(2**(i)),
                        np.kron(sigma_x, np.identity(2**(n-i-1)))
                    )

                    # reset the ith ancilla qubit using the reset gate
                    reset_state = np.dot(reset_gate, reset_state)

    return reset_state


def remove_small_values(logical_state, tolerance=1e-15):
    """
    Used to remove the smaller values from state representation

    * logical_state: The vector state representation of your full qubit system
    """
    # How many total qubits are in our vector representation
    n = int(np.log(len(logical_state))/np.log(2))

    x=0
    for j in range(len(logical_state)):
        if (abs(logical_state[j]) > tolerance):
            # initialize the vector that will hold the single non-zero value in
            # the proper spot
            value_position = np.zeros((1,2**n), dtype=complex)
            # insert the non-zero value in the correct spot
            value_position[:,j] = logical_state[j]
            # Add the value position vector to an array of all the error places
            if x == 0:
                all_vector_states = value_position
            else:
                all_vector_states = \
                    np.append(all_vector_states, value_position , axis=0)
            x+=1

    # find the number of rows and columns in the all error state array so that
    # we can loop over the rows later
    num_rows, num_cols = all_vector_states.shape

    # combine the vector states again
    corrected_vector_state = np.zeros((2**(n),), dtype=complex)
    for j in range(num_rows):
        corrected_vector_state = \
            corrected_vector_state + all_vector_states[j][:]

    return corrected_vector_state



def CNOT_gate_tot(control, target):
    """
    Find out how many operations your CNOT gate is if it is line connected
    """
    if target < control:
        tot_gates = np.abs(2*((control - target) + (control - target - 1)))
    elif control < target:
        tot_gates = np.abs(2*((target - control) + (target - control - 1)))

    return tot_gates


def collapse_dm(rho):
    """
    Used to fully collapse the density matrix when measuring it

    * rho: The density matrix of your system
    """
    # Create Measurement operators for density matrix of N qubits
    for i in range(len(rho)):
        operator = np.zeros((len(rho), len(rho)))
        operator[i][i] = 1
        if i == 0:
            meas_operators = np.array([operator])
        else:
            meas_operators = np.append(meas_operators, [operator], axis = 0)

    # Measure probability of each measurement operator
    meas_probs = np.array([])
    for i in range(len(meas_operators)):
        prob = np.trace(
            np.dot(meas_operators[i].conj().T, np.dot(meas_operators[i], rho))
        ).real   # GP: diagonals should be real -- enforce to suppress warning
        meas_probs = np.append(meas_probs, prob)

    # find which measurement operator is measured based on their probabilities
    # GP: pretty sure these two lines are bugged - will not work for the case
    # -- where all meas probs are the same, e.g. [0.25,0.25,0.25,0.25] - you
    # -- will always get index 0, and it should be an even pick across the
    # -- values.
    # index = random.choices(meas_probs, weights=meas_probs, k=1)
    # index = np.where(meas_probs == index)[0][0]
    index = random.choices(list(range(len(rho))), weights=meas_probs, k=1)[0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(
        meas_operators[index], np.dot(rho, meas_operators[index].conj().T)
    )/(meas_probs[index])

    # Now that we have completed our measurement we are in a pure state.

    # Thus we can find the elemnts on the diagaonal as our final psi since
    # rho = |psi><psi|
    final_psi = np.array([])
    for i in range(len(rho_prime)): # pylint: disable=consider-using-enumerate
        final_psi = np.append(final_psi, rho_prime[i][i])

    return final_psi
