# This file contains different universal error operations that can be used on any number of qubits.

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *

### Applies a random X rotation to one of the physical qubits in your system (randomly) ### 
def random_qubit_x_error(logical_state, qubit_range = None):
    # logical_state: The logical state of the three qubit system you wish to apply the error to
    # qubit_range: The indices you want to consider in your error application (starts at 0)
    
    # total number of qubits in your system
    n = int(np.log(len(logical_state))/np.log(2))
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,n-1)
    if qubit_range != None:
        if error_index != -1:
            error_index = random.randint(qubit_range[0], qubit_range[1])
             
    # Apply the error depending on the index
    if error_index == -1:
        errored_logical_state = logical_state
    else:   
        error_gate = np.kron(np.identity(2**(error_index)), np.kron(sigma_x, np.identity(2**(n-error_index-1))))
        errored_logical_state = np.dot(error_gate, logical_state)

    return errored_logical_state, error_index

### Applies a random Z rotation to one of the physical qubits in your system (randomly) ### 
def random_qubit_z_error(logical_state, qubit_range = None):
    # logical_state: The logical state of the three qubit system you wish to apply the error to
    # qubit_range: The indices you want to consider in your error application (starts at 0)
   
    # total number of qubits in your system
    n = int(np.log(len(logical_state))/np.log(2))
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,n-1)
    if qubit_range != None:
        if error_index != -1:
            error_index = random.randint(qubit_range[0], qubit_range[1])
             
    # Apply the error depending on the index
    if error_index == -1:
        errored_logical_state = logical_state
    else:   
        error_gate = np.kron(np.identity(2**(error_index)), np.kron(sigma_z, np.identity(2**(n-error_index-1))))
        errored_logical_state = np.dot(error_gate, logical_state)

    return errored_logical_state, error_index


### - - - - - - - - - - Gates which contain probability for errors (line connectivity) - - - - - - - - - - ###

### Takes the density matrix after a perfect operation and applies an error gate based on probability of error ###
def gate_error(rho, error_prob, index, n):
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


### - - - CNOT Gates - - - ###

### Apply an adjacent CNOT gate between 2 qubits in a system with line connectivity and errors ###
def errored_adj_CNOT(rho, control, target, qubit_error_probs):
    # rho: the desnity matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    
    gate = np.kron(np.identity(2**(n1)), np.kron(cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    # apply our error gate and find the new density matrix
    error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target], target, tot_qubits)
                    
    return error_rho


### Apply a non-adjacent CNOT gate between 2 qubits in a system with line connectivity and errors ###
def errored_non_adj_CNOT(rho, control, target, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = target - control # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits)
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits)
                
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target-j-2+1], target-j-2+1, tot_qubits)
            
    return error_rho # returns the density matrix of your system


### Apply an adjacent flipped CNOT gate between 2 qubits in a system with line connectivity and errors ###
def errored_flipped_adj_CNOT(rho, control, target, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)

    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # Adds the dimensions needed depending on the tot_qubits
    n1 = target # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    gate = np.kron(np.identity(2**(n1)), np.kron(flipped_cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    # apply our error gate and find the new density matrix
    error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target], target, tot_qubits)
                    
    return error_rho


### Apply a non-adjacent flipped CNOT gate between 2 qubits in a system with line connectivity and errors ###
def errored_flipped_non_adj_CNOT(rho, control, target, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = np.abs(target - control) # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

    # Adds the dimensions needed depending on the tot_qubits
    n1 = target # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(p-j-1)), np.kron(flipped_cnot, np.identity(2**(j))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits)
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits)
                   
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target+j], target+j, tot_qubits)
            
         
    return error_rho # returns the density matrix of your system

### Implement an errored CNOT gate between 2 qubits depending on your control and target qubit
def line_errored_CNOT(state, control, target, qubit_error_probs, form = 'psi'):
    # state: the vector state representation or density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state
        
    # First check if it is a normal CNOT or a flipped CNOT gate
    if control < target:
        # Check if adjacent
        if target - control == 1:
            final_rho = errored_adj_CNOT(rho, control, target, qubit_error_probs)
        else:
            final_rho = errored_non_adj_CNOT(rho, control, target, qubit_error_probs)
    
    #Check if it is a normal CNOT or a flipped CNOT gate
    elif control > target:
        # Check if adjacent
        if control - target == 1:
            final_rho = errored_flipped_adj_CNOT(rho, control, target, qubit_error_probs)
        else:
            final_rho = errored_flipped_non_adj_CNOT(rho, control, target, qubit_error_probs)
    
    return final_rho # output is always the density matrix after the operation


### - - - - - CZ Gates - - - - - ###

### Implement a CZ gate between 2 adjacent qubits in a system ###
def errored_adj_CZ(rho, control, target, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        n1 = control # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    else:
        n1 = target # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system

    gate = np.kron(np.identity(2**(n1)), np.kron(cz, np.identity(2**(n2))))
    # remove small values
    gate[np.abs(gate) < 1e-15] = 0
    
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    
    # apply our error gate and find the new density matrix
    error_rho = gate_error(perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits)

    return error_rho


### Implement a non-adjacent CZ gate between 2 qubits in a system ###
def errored_non_adj_CZ(rho, control, target, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
      # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)

    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = np.abs(target - control) # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

     # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        n1 = control # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system

    else:
        n1 = target # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system

    # Find the correct Hadamard gate to apply so that you convert the CNOT to a CZ
    h_gate = np.kron(np.identity(2**(n1)), np.kron(np.kron(
            np.identity(2**(np.abs(target - control))), hadamard), np.identity(2**(n2))))
    
    # apply the hadamard first to take it to the (+, -) basis
    rho = np.dot(h_gate, np.dot(rho, h_gate.conj().T))
    rho = rad_error(rho, t1, t2, tg)
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # remove small values
                gate[np.abs(gate) < 1e-15] = 0
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits)
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # remove small values
                gate[np.abs(gate) < 1e-15] = 0
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits)
                
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # remove small values
            gate[np.abs(gate) < 1e-15] = 0
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target-j-2+1], target-j-2+1, tot_qubits)
            
    
    # Calculate the final rho
    error_rho = np.dot(h_gate, np.dot(error_rho, h_gate.conj().T))
    
    error_rho = gate_error(error_rho, qubit_error_probs[target], target, tot_qubits)
                
    return error_rho # returns the density matrix of your system


### Implement a CNOT gate between 2 qubits depending on your control and target qubit
def line_errored_CZ(state, control, target, qubit_error_probs, form = 'psi'):
    # state: the vector state representation or density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state

    # Check if adjacent
    if np.abs(target - control) == 1:
        final_rho = errored_adj_CZ(rho, control, target, qubit_error_probs)
    else:
        final_rho = errored_non_adj_CZ(rho, control, target, qubit_error_probs)
    
    return final_rho # output is always the density matrix after the operation



### - - - - - - - - - Gates which contain rad (relaxation and dephasing) errors (line connectivity) - - - - - - - - - ###

### - - - CNOT GATES - - - ###

### Takes the density matrix after a perfect operation and applies an error gate based on t1, t2, and tg ###
def rad_error(rho, t1, t2, tg):
    # rho: density matrix of qubit system after perfect gate was applied
    # t1: the relaxation time of the qubits
    # t2: the dephasing time of the qubits
    # tg: time of the gate you are applying
    
    zero = np.array([1, 0])
    one = np.array([0, 1])
    
    # index: index of qubit that gate was applied (target qubit in this case) 
    ### - can add this, but for now all have same t1 and t2
    
    # total number of qubits in your system
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p_t1 = np.exp(-tg/t1) # find the probability of relaxation
    p_t2 = np.exp(-tg/t2) # find the probability of dephasing
    p_reset = 1 - p_t1 # find the probability of resetting to equilibrium
    
    
    # find the dephasing (phase flip) gate operation
    p_z = (1-p_reset) * (1- (p_t2/p_t1)) * 0.5
    k_z = np.sqrt(p_z) * sigma_z

    # find the relaxation/thermal decay gate operation
    k_reset = np.sqrt(p_reset) * np.kron(zero, zero[np.newaxis].conj().T)

    # find the identity transformation gate operation
    p_I = 1 - p_z - p_reset
    k_I = np.sqrt(p_I) * sigma_I

    # apply the same error to all of our qubits in our system
    for i in range(tot_qubits):
        if i == 0:
            z_gate = k_z
            reset_gate = k_reset
            I_gate = k_I
        else:
            z_gate = np.kron(z_gate, k_z)
            reset_gate = np.kron(reset_gate, k_reset)
            I_gate = np.kron(I_gate, k_I)
            
    # find the density matrix with the 3 types of error gates we found
    final_rho = np.dot(z_gate, np.dot(rho, z_gate.conj().T)) + np.dot(
        reset_gate, np.dot(rho, reset_gate.conj().T)) + np.dot(
        I_gate, np.dot(rho, I_gate.conj().T))

    return final_rho


### Apply an adjacent CNOT gate between 2 qubits in a system with line connectivity and rad errors ###
def rad_adj_CNOT(rho, control, target, t1, t2, tg):
    # rho: the desnity matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    
    gate = np.kron(np.identity(2**(n1)), np.kron(cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    # apply our error gate and find the new density matrix
    error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
                    
    return error_rho


### Apply a non-adjacent CNOT gate between 2 qubits in a system with line connectivity and rad errors ###
def rad_non_adj_CNOT(rho, control, target, t1, t2, tg):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = target - control # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
                
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
            
    return error_rho # returns the density matrix of your system


### Apply an adjacent flipped CNOT gate between 2 qubits in a system with line connectivity and rad errors ###
def rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)

    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # Adds the dimensions needed depending on the tot_qubits
    n1 = target # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    gate = np.kron(np.identity(2**(n1)), np.kron(flipped_cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    # apply our error gate and find the new density matrix
    error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
                    
    return error_rho


### Apply a non-adjacent flipped CNOT gate between 2 qubits in a system with line connectivity and rad errors ###
def rad_flipped_non_adj_CNOT(rho, control, target, t1, t2, tg):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = np.abs(target - control) # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

    # Adds the dimensions needed depending on the tot_qubits
    n1 = target # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(p-j-1)), np.kron(flipped_cnot, np.identity(2**(j))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
                   
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
            
         
    return error_rho # returns the density matrix of your system

### Implement a rad CNOT gate between 2 qubits depending on your control and target qubit
def line_rad_CNOT(state, control, target, t1, t2, tg, form = 'psi'):
    # state: the vector state representation or density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state
        
    # First check if it is a normal CNOT or a flipped CNOT gate
    if control < target:
        # Check if adjacent
        if target - control == 1:
            final_rho = rad_adj_CNOT(rho, control, target, t1, t2, tg)
        else:
            final_rho = rad_non_adj_CNOT(rho, control, target, t1, t2, tg)
    
    #Check if it is a normal CNOT or a flipped CNOT gate
    elif control > target:
        # Check if adjacent
        if control - target == 1:
            final_rho = rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg)
        else:
            final_rho = rad_flipped_non_adj_CNOT(rho, control, target, t1, t2, tg)
    
    return final_rho # output is always the density matrix after the operation

### - - - CZ GATES - - - ###

### Implement a rad CZ gate between 2 adjacent qubits in a system ###
def rad_adj_CZ(rho, control, target, t1, t2, tg):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        n1 = control # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    else:
        n1 = target # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system

    gate = np.kron(np.identity(2**(n1)), np.kron(cz, np.identity(2**(n2))))
    
    # remove small values
    gate[np.abs(gate) < 1e-15] = 0        
    
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    
    # apply our error gate and find the new density matrix
    error_rho = rad_error(perfect_gate_rho, t1, t2, tg)

    return error_rho


### Implement a non-adjacent rad CZ gate between 2 qubits in a system ###
def rad_non_adj_CZ(rho, control, target, t1, t2, tg):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    
      # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)

    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = np.abs(target - control) # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

     # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        n1 = control # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system

    else:
        n1 = target # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Find the correct Hadamard gate to apply so that you convert the CNOT to a CZ
    h_gate = np.kron(np.identity(2**(n1)), np.kron(np.kron(
            np.identity(2**(np.abs(target - control))), hadamard), np.identity(2**(n2))))
    
    # apply the hadamard first to take it to the (+, -) basis
    rho = np.dot(h_gate, np.dot(rho, h_gate.conj().T))
    rho = rad_error(rho, t1, t2, tg)

    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = rad_error(perfect_gate_rho, t1, t2, tg)

            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = rad_error(perfect_gate_rho, t1, t2, tg)

        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = rad_error(perfect_gate_rho, t1, t2, tg)
            

    # Calculate the final rho by applying the h gate again to bring it back to the (0, 1) basis
    error_rho = np.dot(h_gate, np.dot(error_rho, h_gate.conj().T))
    error_rho = rad_error(error_rho, t1, t2, tg)
    
    return error_rho # returns the density matrix of your system


### Implement a CNOT gate between 2 qubits depending on your control and target qubit
def line_rad_CZ(state, control, target, t1, t2, tg, form = 'psi'):
    # state: the vector state representation or density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state

    # Check if adjacent
    if np.abs(target - control) == 1:
        final_rho = rad_adj_CZ(rho, control, target, t1, t2, tg)
    else:
        final_rho = rad_non_adj_CZ(rho, control, target, t1, t2, tg)
    
    return final_rho # output is always the density matrix after the operation

### - - - - - Gates which contain gate errors and rad (relaxation and dephasing) errors (line connectivity) - - - - - ###

### - - - CNOT GATES - - - ###

### Apply a adjacent CNOT gate between 2 qubits in system with line connectivity (probability of error and rad errors) ###
def prob_rad_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs):
    # rho: the desnity matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    
    gate = np.kron(np.identity(2**(n1)), np.kron(cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    # apply our error gates and find the new density matrix
    error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target], target, tot_qubits) # prob error
    error_rho = rad_error(error_rho, t1, t2, tg) # rad error

                    
    return error_rho


### Apply a non-adjacent CNOT gate between 2 qubits in system with line connectivity (probability of error and rad errors) ###
def prob_rad_non_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system

    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = target - control # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

    # Adds the dimensions needed depending on the tot_qubits
    n1 = control # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits) # prob error
                error_rho = rad_error(error_rho, t1, t2, tg) # rad error
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits) # prob error
                error_rho = rad_error(error_rho, t1, t2, tg) # rad error
                
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            error_rho = gate_error(
                perfect_gate_rho, qubit_error_probs[target-j-2+1], target-j-2+1, tot_qubits) # prob error
            error_rho = rad_error(error_rho, t1, t2, tg) # rad error
            
    return error_rho # returns the density matrix of your system


### Apply a adjacent flipped CNOT gate between 2 qubits in system with line connectivity (probability of error and rad errors) ###
def prob_rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)

    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # Adds the dimensions needed depending on the tot_qubits
    n1 = target # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    gate = np.kron(np.identity(2**(n1)), np.kron(flipped_cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    # apply our error gate and find the new density matrix
    error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target], target, tot_qubits) # prob error
    error_rho = rad_error(error_rho, t1, t2, tg) # rad error 
                    
    return error_rho


### Apply a non-adjacent flipped CNOT gate between 2 qubits in system with line connectivity (probability of error and rad errors) ###
def prob_rad_flipped_non_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = np.abs(target - control) # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

    # Adds the dimensions needed depending on the tot_qubits
    n1 = target # exponent used to tensor the left side identity matrix for our full system
    n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(p-j-1)), np.kron(flipped_cnot, np.identity(2**(j))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gates and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits) # prob error
                error_rho = rad_error(error_rho, t1, t2, tg) # rad error
                
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gates and find the new density matrix
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits) # prob error
                error_rho = rad_error(error_rho, t1, t2, tg) # rad error
                   
        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gates and find the new density matrix
            error_rho = gate_error(
                perfect_gate_rho, qubit_error_probs[target+j], target+j, tot_qubits) # prob error
            error_rho = rad_error(error_rho, t1, t2, tg) # rad error
            
         
    return error_rho # returns the density matrix of your system
            
            
### Implement a CNOT gate between 2 qubits in system with line connectivity (probability of error and rad errors) ###
def prob_line_rad_CNOT(state, control, target, t1, t2, tg, qubit_error_probs, form = 'psi'):
    # state: the vector state representation or density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state
        
    # First check if it is a normal CNOT or a flipped CNOT gate
    if control < target:
        # Check if adjacent
        if target - control == 1:
            final_rho = prob_rad_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs)
        else:
            final_rho = prob_rad_non_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs)
    
    #Check if it is a normal CNOT or a flipped CNOT gate
    elif control > target:
        # Check if adjacent
        if control - target == 1:
            final_rho = prob_rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs)
        else:
            final_rho = prob_rad_flipped_non_adj_CNOT(rho, control, target, t1, t2, tg, qubit_error_probs)
    
    return final_rho # output is always the density matrix after the operation

### - - - CZ GATES - - - ###

### Implement a rad CZ gate between 2 adjacent qubits in a system ###
def prob_rad_adj_CZ(rho, control, target, t1, t2, tg, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        n1 = control # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system
    else:
        n1 = target # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system

    gate = np.kron(np.identity(2**(n1)), np.kron(cz, np.identity(2**(n2))))
    # remove small values
    gate[np.abs(gate) < 1e-15] = 0
    perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
    
    # apply our error gate and find the new density matrix
    error_rho = gate_error(perfect_gate_rho, qubit_error_probs[target], target, tot_qubits) # prob error
    error_rho = rad_error(error_rho, t1, t2, tg) # rad error

    return error_rho


### Implement a non-adjacent rad CZ gate between 2 qubits in a system ###
def prob_rad_non_adj_CZ(rho, control, target, t1, t2, tg, qubit_error_probs):
    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be a larger index than control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    
      # find our density matrix
#     rho = np.kron(psi, psi[np.newaxis].conj().T)

    # How many total qubits are in our vector representation
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    p = np.abs(target - control) # used to index over all gates neeeded to compose final gate
    all_dots = np.array([[]]) # array used to keep track of the components we will combine at the end

     # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        n1 = control # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - target - 1 # exponent used to tensor the right side identity matrix for our full system

    else:
        n1 = target # exponent used to tensor the left side identity matrix for our full system
        n2 = tot_qubits - control - 1 # exponent used to tensor the right side identity matrix for our full system

        
    # Find the correct Hadamard gate to apply so that you convert the CNOT to a CZ
    h_gate = np.kron(np.identity(2**(n1)), np.kron(np.kron(
            np.identity(2**(np.abs(target - control))), hadamard), np.identity(2**(n2))))
    
    # apply the hadamard first to take it to the (+, -) basis
    rho = np.dot(h_gate, np.dot(rho, h_gate.conj().T))
    rho = gate_error(rho, qubit_error_probs[target], target, tot_qubits) # depolarizing error
    rho = rad_error(rho, t1, t2, tg) # rad error
    
    # Applies the gates twice (square in our formula)
    for k in range(0,2):
        if k != 0:
            rho = error_rho
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                if control < target:
                    error_rho = gate_error(
                        perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits) # prob_error
                else:
                    error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits) # prob error
                
                error_rho = rad_error(error_rho, t1, t2, tg) # rad error

            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds the perfect gate to an array
                gate = all_dots[j] # sets the current gate
                # applies the perfect gate to our density matrix
                perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
                # apply our error gate and find the new density matrix
                if control < target:
                    error_rho = gate_error(
                        perfect_gate_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits) # prob_error
                else:
                    error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits) # prob error
                
                error_rho = rad_error(error_rho, t1, t2, tg) # rad error

        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            perfect_gate_rho = np.dot(gate, np.dot(error_rho, gate.conj().T)) 
            # apply our error gate and find the new density matrix
            if control < target:  
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[target-j-2+1], target-j-2+1, tot_qubits) # prob error
            else:
                error_rho = gate_error(
                    perfect_gate_rho, qubit_error_probs[target+j], target+j, tot_qubits) # prob error
                
            error_rho = rad_error(error_rho, t1, t2, tg) # rad error
            

    # Calculate the final rho
    error_rho = np.dot(h_gate, np.dot(error_rho, h_gate.conj().T))
    error_rho = gate_error(error_rho, qubit_error_probs[target], target, tot_qubits) # prob error
    error_rho = rad_error(error_rho, t1, t2, tg) # rad error
    
    return error_rho # returns the density matrix of your system

   
### Implement a CNOT gate between 2 qubits depending on your control and target qubit
def prob_line_rad_CZ(state, control, target, t1, t2, tg, qubit_error_probs, form = 'psi'):
    # state: the vector state representation or density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state

    # Check if adjacent
    if np.abs(target - control) == 1:
        final_rho = prob_rad_adj_CZ(rho, control, target, t1, t2, tg, qubit_error_probs)
    else:
        final_rho = prob_rad_non_adj_CZ(rho, control, target, t1, t2, tg, qubit_error_probs)
    
    return final_rho # output is always the density matrix after the operation


### - - - - - - - - - SPAM error function - - - - - - - - - ###

### Takes the density matrix after state preparation and before measurement and applies an error gate with probability ###
def spam_error(rho, error_prob, index):
    # rho: density matrix of qubit system after perfect gate was applied
    # error_prob: probability for state preparation error or measuremnt error
    # index: index of qubit that was prepared/measureed
    
    # total number of qubits in your system
    tot_qubits = int(np.log(len(rho))/np.log(2))
    
    # qubit error rates:
    KM0 = np.sqrt(1-error_prob) * sigma_I
    KM1 = np.sqrt(error_prob) * sigma_x
    
    # qubit error gates
    KM0 = np.kron(np.identity(2**(index)), np.kron(KM0, np.identity(2**(tot_qubits-index-1))))
    KM1 = np.kron(np.identity(2**(index)), np.kron(KM1, np.identity(2**(tot_qubits-index-1)))) 
    
    # apply error gates 
    spam_rho = np.dot(KM0, np.dot(rho, KM0.conj().T)) + np.dot(KM1, np.dot(rho, KM1.conj().T))
    
    return spam_rho


