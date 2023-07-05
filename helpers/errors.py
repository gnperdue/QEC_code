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
    
    
### Takes the density matrix after a perfect operation and applies an error gate based on probability of error ###
def qubit_gate_error_matrix(rho, error_prob, index, n):
    # rho: density matrix of qubit system after perfect gate was applied
    # error_prob: probability for gate operation error
    # index: index of qubit that gate was applied (target qubit in this case)
    # n: total number of qubits in your system
    
    # qubit error rates:
    KD0 = np.sqrt(1-error_prob) * sigma_I
    KD1 = np.sqrt(error_prob/3) * sigma_x
    KD2 = np.sqrt(error_prob/3) * sigma_z
    KD3 = np.sqrt(error_prob/3) * sigma_y
    
    # qubit error gates
    KD0 = np.kron(np.identity(2**(index)), np.kron(KD0, np.identity(2**(n-index-1))))
    KD1 = np.kron(np.identity(2**(index)), np.kron(KD1, np.identity(2**(n-index-1)))) 
    KD2 = np.kron(np.identity(2**(index)), np.kron(KD2, np.identity(2**(n-index-1)))) 
    KD3 = np.kron(np.identity(2**(index)), np.kron(KD3, np.identity(2**(n-index-1))))
    
    # apply error gates (qubit 0 and qubit 2 will not be affected by error gates, although we do apply Identity to q0)
    D_rho = np.dot(KD0, np.dot(rho, KD0.conj().T)) + np.dot(
        KD1, np.dot(rho, KD1.conj().T)) + np.dot(
        KD2, np.dot(rho, KD2.conj().T)) + np.dot(
        KD3, np.dot(rho, KD3.conj().T))
    
    return D_rho