# This file contains different universal error operations that can be used on any number of qubits.

import numpy as np
import random
from helpers.qec_helpers import *
from helpers.gates import *

### Applies a random X rotation to one of the physical qubits in your system (randomly) ### 
def random_qubit_x_error(logical_state, qubit_range = None):
    # logical_state: The logical state of the three qubit system you wish to apply the error to
    
    # total number of qubits in your system
    n = int(np.log(len(logical_state))/np.log(2))
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,n-1)
    if qubit_range != None:
        if error_index != -1:
            error_index = random.randint(qubit_range[0], qubit_range[1]-1)
             
    # Apply the errro depending on the index
    if error_index == -1:
        errored_logical_state = logical_state
        return errored_logical_state, error_index
    else:   
        error_gate = np.kron(np.identity(2**(error_index)), np.kron(sigma_x, np.identity(2**(n-error_index-1))))
        errored_logical_state = np.dot(error_gate, logical_state)

        return errored_logical_state, error_index

### Applies a random Z rotation to one of the physical qubits in your system (randomly) ### 
def random_qubit_z_error(logical_state, qubit_range = None):
    # logical_state: The logical state of the three qubit system you wish to apply the error to
    
    # total number of qubits in your system
    n = int(np.log(len(logical_state))/np.log(2))
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,n-1)
    if qubit_range != None:
        if error_index != -1:
            error_index = random.randint(qubit_range[0], qubit_range[1]-1)
             
    # Apply the errro depending on the index
    if error_index == -1:
        errored_logical_state = logical_state
        return errored_logical_state, error_index
    else:   
        error_gate = np.kron(np.identity(2**(error_index)), np.kron(sigma_z, np.identity(2**(n-error_index-1))))
        errored_logical_state = np.dot(error_gate, logical_state)

        return errored_logical_state, error_index