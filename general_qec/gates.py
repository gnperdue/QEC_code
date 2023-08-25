"""
This file contains functions that implement different types of useful gates to
the circuit.
"""
import numpy as np
import random

### Pauli operators
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,1j],[-1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_I = np.identity(2)

### Hadamard Gate
hadamard = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]])

### - - - - - - - - - - - CNOT GATES - - - - - - - - - - - ###

### CNOT gate
cnot = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])

# flips the roles of control and target in our usual CNOT gate
flipped_cnot = np.array([[1, 0, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])
            
 
def adj_CNOT(control, target, tot_qubits):
    """
    Implement a CNOT gate between 2 adjacent qubits in a system
    
    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must > control)
    tot_qubits: total number of qubits in the system

    target - control != 1
    """   
    assert target - control == 1, "target - control != 1"
    # exponent used to tensor the left side identity matrix for our full system
    n1 = control
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - target - 1    
    final_gate = np.kron(
        np.identity(2**(n1)), np.kron(cnot, np.identity(2**(n2)))
    )
    
    return final_gate


def flipped_adj_CNOT(control, target, tot_qubits):
    """
    Implement a flipped CNOT gate between 2 adjacent qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must < control)
    tot_qubits: total number of qubits in the system

    control - target != 1
    """
    assert control - target == 1, "control - target != 1"
    # exponent used to tensor the left side identity matrix for our full system
    n1 = target
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - control - 1
    final_gate = np.kron(
        np.identity(2**(n1)), np.kron(flipped_cnot, np.identity(2**(n2)))
    )

    return final_gate


def small_non_adj_CNOT():
    """
    Used to quickly perform a CNOT gate on 2 non-adjacent qubits (i.e. |psi> 
    and |q_1>) --- for 3 qubits
    """
    cnot = np.identity(8)
    cnot[4:] = 0
    cnot[4][5] = 1
    cnot[5][4] = 1
    cnot[7][6] = 1
    cnot[6][7] = 1
    return cnot

def non_adj_CNOT(control, target, tot_qubits):
    """
    Implement a non-adjacent CNOT gate between 2 qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must be > control)
    tot_qubits: total number of qubits in the system

    (target - control) must be greater than 1
    """
    assert target - control > 1, "(target - control) must be greater than 1"
    # used to index over all gates neeeded to compose final gate
    p = target - control
    # array used to keep track of the components we will combine at the end
    all_dots = np.array([[]])
    
    # Indexing over the values of p to get the first half of the formula
    for j in range(p):
        # Sets the next component of the matrix multiplication up
        next_dot = np.kron(
            np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1)))
        )
        # Adds the components to the array and multiplies them together
        if j == 0:
            all_dots = np.array([next_dot])
            gate = all_dots[j]
        else:
            all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = np.dot(gate, all_dots[j])
            
    # Indexing over values of p such that we get the 2nd half of the equation
    # together
    for j in range(p - 2):
        gate = np.dot(gate, all_dots[p-j-2])
    
    # Squares the final matrix
    final_gate = np.dot(gate, gate)
    
    # exponent used to tensor the left side identity matrix for our full system
    n1 = control
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - target - 1
    final_total_gate = np.kron(
        np.identity(2**(n1)), np.kron(final_gate, np.identity(2**(n2)))
    )
    return final_total_gate

def flipped_non_adj_CNOT(control, target, tot_qubits):
    """
    Implement a flipped non-adjacent CNOT gate between 2 qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must be < control)
    tot_qubits: total number of qubits in the system

    (control - target) must be greater than 1
    """
    assert control - target > 1, "(control - target) must be greater than 1"
    # used to index over all gates neeeded to compose final gate
    p = np.abs(target - control)
    # array used to keep track of the components we will combine at the end
    all_dots = np.array([[]])
    
    # Indexing over the values of p to get the first half of the formula
    for j in range(p):
        # Sets the next component of the matrix multiplication up
        next_dot = np.kron(
            np.identity(2**(p-j-1)), np.kron(flipped_cnot, np.identity(2**(j)))
        )
        
        # Adds the components to the array and multiplies them together
        if j == 0:
            all_dots = np.array([next_dot])
            gate = all_dots[j]
        else:
            all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = np.dot(gate, all_dots[j])
            
    # Indexing over values of p such that we get the 2nd half of the equation
    # together
    for j in range(p - 2):
        gate = np.dot(gate, all_dots[p-j-2])
    
    # Squares the final matrix
    final_gate = np.dot(gate, gate)
    
    # exponent used to tensor the left side identity matrix for our full system
    n1 = target
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - control - 1
    final_total_gate = np.kron(
        np.identity(2**(n1)), np.kron(final_gate, np.identity(2**(n2)))
    )
    
    return final_total_gate


def CNOT(control, target, tot_qubits):
    """
    Implement a CNOT gate between 2 qubits depending on your control and
    target qubit.
    """
    
    # First check if it is a normal CNOT or a flipped CNOT gate
    if control < target:
        # Check if adjacent
        if target - control == 1:
            gate = adj_CNOT(control, target, tot_qubits)
        else:
            gate = non_adj_CNOT(control, target, tot_qubits)
    
    #Check if it is a normal CNOT or a flipped CNOT gate
    elif control > target:
        # Check if adjacent
        if control - target == 1:
            gate = flipped_adj_CNOT(control, target, tot_qubits)
        else:
            gate = flipped_non_adj_CNOT(control, target, tot_qubits)
    
    return gate


### - - - - - - - - - - - C-Z GATES - - - - - - - - - - - ###

# Note that we will not need to impliment a flipped CZ gate becuase the logic
# table is the same for both.

### Control-Z gate
# cz = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, -1]])
## OR could also use:
cz = np.dot(
    np.kron(np.identity(2), hadamard),
    np.dot(cnot, np.kron(np.identity(2), hadamard))).round().astype(int)


def adj_CZ(control, target, tot_qubits):
    """
    Implement a CZ gate between 2 adjacent qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must != control)
    tot_qubits: total number of qubits in the system
    """
    assert target != control, "target must not equal control"
    # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        # exponent used to tensor the left side iden matrix for our full system
        n1 = control
        # exponent used to tensor the right side iden matrix for full system
        n2 = tot_qubits - target - 1
    else:
        # exponent used to tensor the left side iden matrix for our full system
        n1 = target
        # exponent used to tensor the right side iden matrix for full system
        n2 = tot_qubits - control - 1

    final_gate = np.kron(
        np.identity(2**(n1)), np.kron(cz, np.identity(2**(n2)))
    )
    
    # remove small values
    final_gate[np.abs(final_gate) < 1e-15] = 0
    
    return final_gate


def non_adj_CZ(control, target, tot_qubits):
    """
    Implement a non-adjacent CZ gate between 2 qubits in a system ###

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must != control)
    tot_qubits: total number of qubits in the system
    """
    assert target != control, "target must not equal control"
    # used to index over all gates neeeded to compose final gate
    p = np.abs(target - control)
    # array used to keep track of the components we will combine at the end
    all_dots = np.array([[]])
    # Indexing over the values of p to get the first half of the formula
    for j in range(p):
        # Sets the next component of the matrix multiplication up
        next_dot = np.kron(
            np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1)))
        )

        # Adds the components to the array and multiplies them together
        if j == 0:
            all_dots = np.array([next_dot])
            gate = all_dots[j]
        else:
            all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = np.dot(gate, all_dots[j])

    # Indexing over values of p such that we get the 2nd half of the equation
    # together
    for j in range(p - 2):
        gate = np.dot(gate, all_dots[p-j-2])

    # Squares the final matrix
    final_gate = np.dot(gate, gate)

    # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        # exponent used to tensor the left side iden matrix for our full system
        n1 = control
        # exponent used to tensor the right side iden matrix for our system
        n2 = tot_qubits - target - 1
    else:
        # exponent used to tensor the left side iden matrix for our full system
        n1 = target
        # exponent used to tensor the right side identity matrix for our system
        n2 = tot_qubits - control - 1

    # Find the correct Hadamard gate to apply so that you convert the CNOT to
    # a CZ.
    h_gate = np.kron(
        np.identity(2**(n1)),
        np.kron(
            np.kron(np.identity(2**(np.abs(target - control))), hadamard), 
            np.identity(2**(n2))
        )
    )

    # Calculate the final gate
    final_total_gate = np.dot(
        h_gate, np.dot(
            np.kron(
                np.identity(2**(n1)), np.kron(final_gate, np.identity(2**(n2)))
            ),
            h_gate
        )
    )
    
    # remove small values
    final_total_gate[np.abs(final_total_gate) < 1e-15] = 0
    
    return final_total_gate



def CZ(control, target, tot_qubits):
    """
    Implement a Control-Z gate between 2 qubits depending on your parameters
    """
    # Check if adjacent
    if target - control == 1:
        gate = adj_CZ(control, target, tot_qubits)
    else:
        gate = non_adj_CZ(control, target, tot_qubits)
    
    return gate





