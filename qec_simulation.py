import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# using datetime module
import datetime # used to see how long things take here

# For fitting exponentials
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# for exponential warnings
import warnings
#suppress warnings
warnings.filterwarnings('ignore')

from general_qec.qec_helpers import *
from general_qec.errors import *
from circuit_specific.realistic_three_qubit import *
from circuit_specific.realistic_steane import *
from circuit_specific.realistic_ft_steane import *

### Runs the top level simulation user interface and calls on simulation function
def run_sim():
    print('Hello! Welcome to the quantum error correction simulator.')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('Key Notes before beginning the simulation.')
    print('* Some error codes take some time to iterate depending on the parameters that you input.')
    print(' - a. 3-qubit code is the fastest and can usually iterate 1000 times in roughly 25 sec.')
    print(' - b. 7-qubit Steane code usually takes about 5 min per iteration.')
    print(' - c. 7-qubit Fault tolerant Steane code usually takes about 10-15 min per iteration.')
    print(' - d. 9 qubit code has not been tested yet.')
    print('(This code was run using a 2020 MacBook Pro (M1) with 8GB RAM and macOS 13.4.1)')
    print('* For more information on the physics and mathematics behind our simulation, you can visit the implementation knowledge folder.')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    while True:
        print('Please choose the circuit you would like to run: \n(type in the number as displayed in the list)')
        print('\n1. 3-qubit code\n2. 7-qubit Steane code\n3. Fault tolerant 7-qubit Steane code\n4. 9-qubit code')
        while True:
            try:
                circuit = int(input('Selection: '))
                if 1 <= circuit <=4:
                    break
                else:
                    print('Please input a valid circuit value.')
            except ValueError:
                print("Oops!  That was not a valid value.  Try again...")

        print('- - - - - - - - - -')

        print('In our simulation we will initialize the logical state as |1>.')
        state_bool = bool(input('Would you like to input your own initial state? (Leave blank if not)'))
        if state_bool:
            print('...')
            print('We represent our initial state as alpha*|0> + beta*|1> where |alpha|^2 + |beta|^2 = 1')
            while True:
                try:
                    alpha = float(input('\nalpha: '))
                    beta = float(input('\nbeta: '))
                    if round(np.abs(alpha)**2 + np.abs(beta)**2, 3) == 1:
                        break
                    else:
                        print("Oops!  Those were not valid values.  Try again...")
                except ValueError:
                    print("Oops!  Those were not valid values.  Try again...")

            psi = np.array([alpha, beta])

        else:
            psi = np.array([0, 1])

        print('Initial state: ', psi[0],'|0> + ',psi[1],'|1>')
        print('- - - - - - - - - -')

        print('We will now select whick errors we would like to implement.')
        print('Enter any value if you wish to include that error. (If not, leave it blank and press \'enter\')')
        dep_bool = bool(input('\n1. Depolarization. Adds some probability that your gate operations are incorrect.  '))
        spam_bool = bool(input('2. SPAM. Adds some probability that you will incorrectly prepare and measure your state.  '))
        rad_bool = bool(input('3. Relaxation and Dephasing. Adds qubit decay and environmental decoherence to your system.  '))

        print('- - - - - - - - - -')

        print('\nNow please imput the parameters of your errors.')
        print('* For more information on how we define each type of error please visit [05. Error Models].\n')
        if dep_bool:
            print('...')
            while True:
                try:
                    dep = float(input('\nDepolarization. error probability: '))
                    if 0 <= dep <= 1:
                        break
                    else:
                        print("Oops!  Value must be less than 1.  Try again...")
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        else:
            dep = None

        if spam_bool:
            print('...')
            while True:
                try:
                    spam_prob = float(input('\nSPAM. probability for state preparation and measurement errors: '))
                    if 0 <= spam_prob <= 1:
                        break
                    else:
                        print("Oops!  Value must be less than 1.  Try again...")
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        else:
            spam_prob = None

        if rad_bool:
            print('...')
            print('\nRelaxation and Dephasing. (For times please use the following format: ae-b or decimal representation)')

            while True:
                try:
                    t1 = float(input('T1. relaxation time of your qubits (sec) [suggested O(e-4)]: '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

            while True:
                try:
                    t2 = float(input('T2. dephasing time of your qubits (sec) [suggested O(e-4)]: '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

            while True:
                try:
                    tg = float(input('Tg. the gate time of all gate operations in the circuit (sec) [suggested O(e-8)]: '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        else:
            t1 = t2 = tg = None

        print('- - - - - - - - - -')

        print('Now we will select how many times you would like to iterate your circuit.')
        print('Remember that the larger circuits may take 5-15 minutes per iteration due to the size.')
        while True:
            try:
                iterations = int(input('\nIterations: '))
                break
            except ValueError:
                print("Oops!  That was not a valid value.  Try again...")

        print('. . . . . . . . . .')
        print('Thank you, we will now output the information of your circuit.')

        while True:
            print('. . . . . . . . . .')

            simulate_qec(circuit, psi, depolarization=dep, spam_prob=spam_prob, t1=t1, t2=t2, tg=tg, iterations=iterations)
            print('\n')
            print('- - - - - - - - - -')
            # Check what they want to do next
            print('What would you like to do next?\n1. Run the same simulation again.\n2. Start over and input different parameters.\n3. Run the simulation many times and create a sampled distribution of data.')
            while True:
                try:
                    selection = int(input('\nSelection: '))
                    if 1 <= selection <=3:
                        break
                    else:
                        print('Please input a valid value.')
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
            # run simulation again
            if selection !=1:
                break
        # start over and input different parameters
        if selection !=2:
            break
        
    print('- - - - - - - - - -')
    print('What would you like to do in this sample distribution?')
    print('1. Check distribution of iteration at which circuit logical state failure occurs.')
    print('2. Check distribution of the logical T1 of your circuit. (initial state will be changed to |1>)')
    while True:
            try:
                selection = int(input('\nSelection: '))
                if 1 <= selection <=2:
                    break
                else:
                    print('Please input a valid value.')
            except ValueError:
                print("Oops!  That was not a valid value.  Try again...")
    
    while True:
        print('How many samples would you like? (remember we will iteraate the circuit many times per sample) ')
        try:           
            samples = int(input())
            break
        except ValueError:
            print("Oops!  That was not a valid value.  Try again...")

    # selected plotting the circuit failure iteration counts
    if selection == 1:
        while True:
            print('...')
            print('Creating distribution iteration at which circuit logical state failure occurs.')
            
            # - Run the simulation that we want - #
            if circuit == 1: # three qubit code
                three_qubit_sample_failure(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
            elif circuit == 2: # Steane code
                steane_sample_failure(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
            elif circuit == 3: # fault tolerant Steane code
                ft_steane_sample_failure(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
            elif circuit == 4: # nine qubit code
                nine_qubit_sample_failure(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
                
            print('- - - - - - - - - -')
            selection = bool(input('Would you like to run again? (leave blank if not)'))
            if not selection:
                break
    # selected plotting t1 time distributions
    elif selection == 2:
        print('...')

        print('We will now create a distribution of the logical T1 of your circuit.')
        print('Remember that the initial state will be changed to |1>.')
        if (t1==None and t2==None and tg==None):
            print('...')
            print('For this simulation we will need you to select physical T1, T2, and gate time (Tg).')
            print('...')
            print('Relaxation and Dephasing. (For times please use the following format: ae-b or decimal representation)')

            while True:
                try:
                    t1 = float(input('T1. relaxation time of your qubits (sec): '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

            while True:
                try:
                    t2 = float(input('T2. dephasing time of your qubits (sec): '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

            while True:
                try:
                    tg = float(input('Tg. the gate time of all gate operations in the circuit (sec): '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        while True:
            print('...')
            print('Creating your distribution histogram for logical T1 of your system...')
            
            # - Run the simulation that we want - #
            if circuit == 1: # three qubit code
                three_qubit_sample_t1(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
            elif circuit == 2: # Steane code
                steane_sample_t1(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
            elif circuit == 3: # fault tolerant Steane code
                ft_steane_sample_t1(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
            elif circuit == 4: # nine qubit code
                nine_qubit_sample_t1(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
                
            print('- - - - - - - - - -')
            selection = bool(input('Would you like to run again? (leave blank if not)'))
            if not selection:
                break
    
    
    
    ### End the simulation
    print('- - - - - - - - - - - - - - - - - - -')
    print('Thank you for using our simulation! To simulate again, go ahead run the run_sim() cell again.')
    print('- - - - - END OF SIMULATION - - - - -')


##### - - - - - FUNCTIONS USED IN THE SIMULATION INTERFACE ABOVE. - - - - - #####

### Choose which circuit we want to run.   
def simulate_qec(circuit, psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # circuit: which circuit do you want to simulate
    # psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    if circuit == 1:
        three_qubit_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
    elif circuit == 2:
        steane_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
    elif circuit == 3:
        ft_steane_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
    elif circuit == 4:
        nine_qubit_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
        
        
        
        
### - - - - - - 3-qubit simulation functions - - - - - - ###

### Run the 3 qubit simulation realistically with paramters and a certain number of iterations.       
def three_qubit_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    qubit_error_probs = np.array([])
    
    ideal_state = np.dot(CNOT(1, 2, 5), np.dot(CNOT(0, 1, 5), np.kron(
        initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero))))))
                          
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(5):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
        
    initialized_rho = initialize_three_qubit_realisitc(
        initial_psi, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

    rho = initialized_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    # all_pops = np.array([])
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = three_qubit_realistic(rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(np.identity(2), np.identity(2)))))
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))

        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(np.identity(2), np.identity(2)))))
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))

        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)
        
        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 5)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = '|000>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = '|111>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running 3 qubit code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    if 0<popt[1]<1:
        plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
        print('- - - - -')
        circuit_runs = 1/popt[1]
        if tg!=None:
            print('Calculated Circuit iterations until logical failure: ', circuit_runs)
            print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
        else:
            print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the state of logical failure for the 3 qubit code
def three_qubit_sample_failure(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on calculating the probability of state measurements overtime...')
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    ideal_state = np.dot(CNOT(1, 2, 5), np.dot(CNOT(0, 1, 5), np.kron(
        initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero))))))
    
    ideal_bits = vector_state_to_bit_state(ideal_state, 5)[0]

    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(5):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
        
    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        # Initialize our logical state depending on parameters
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1 = t1, t2 = t2, tg = tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = three_qubit_realistic(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # Check if we are still in our ideal state
            collapsed_bits = vector_state_to_bit_state(collapse_dm(rho), 5)[0][0]
            if collapsed_bits not in ideal_bits:
                break

        count = np.append(count, i)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')
    
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 5)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
        
    bin_num = int(samples/20) + 5
        
    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    print('... Number of bins:', len(bins)-1, '...')
    
    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the logical T1 of your system over many runs        
def three_qubit_sample_t1(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):        
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    ideal_state = np.dot(CNOT(1, 2, 5), np.dot(CNOT(0, 1, 5), np.kron(
        initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero))))))
    
    ideal_bits = vector_state_to_bit_state(ideal_state, 5)[0]

    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(5):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    t1_times = np.array([])
    for k in range(samples):
        # initialize our logical state
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1 = t1, t2 = t2, tg = tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            count = np.append(count, i)
            # apply circuit
            rho = three_qubit_realistic(rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob) 

            # measure the probability of being in the state |111> from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(np.identity(2), np.identity(2)))))
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))

            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        circuit_t1 = ((circuit_runs * 29 + 2) * tg)*1e6
        t1_times = np.append(t1_times, circuit_t1)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 5)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(t1_times, bins = bins, label = 'Logical T1 Distribution', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1')
    plt.xlabel('Logical T1') 
    plt.ylabel('Number of Samples') 

    # - - - Fitting a curve to our plot - - - #

#     xdata = (bins[1:])[n!=0]
#     ydata = n[n!=0]

#     popt, pcov = curve_fit(exp_decay, xdata, ydata)

#     plt.plot(xdata, exp_decay(xdata, *popt), 'black',
#              label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
#     print('- - -')

#     circuit_runs = 1/popt[1]
#     print('Characteristic number of runs until failure: ', circuit_runs)

#     char_time = (((circuit_runs * 29) + 2) * tg)
#     print('Characteristic time until failure: ', char_time, 'sec')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
        
        
### - - - - - - Steane Code simulation functions - - - - - - ###

### Run the Steane code simulation realistically with paramters and a certain number of iterations.       
def steane_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_steane_logical_state(initial_state)
    
    qubit_error_probs = np.array([])
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(10):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    rho = initial_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = realistic_steane(
        rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        # Measurement operator to see if you are in the 0 logical state
        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3))))))))
        
        # probability of being in the 0 logical state
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))
        
        # Measurement operator to see if you are in the 1 logical state
        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3))))))))
        
        # probability of being in the 1 logical state
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))
        
        # any other probability
        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)
    
        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = 'Logical |0>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = 'Logical |1>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running Steane code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - - - -')
    circuit_runs = 1/popt[1]
    if tg!=None:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
        print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
    else:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the state of logical failure for the Steane code
def steane_sample_failure(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on calculating the probability of state measurements overtime...')

    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 10)[0]
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(10):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        # Initialize our logical state depending on parameters
        rho = realistic_steane(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = realistic_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # Check if we are still in our ideal state
            collapsed_bits = vector_state_to_bit_state(collapse_dm(rho), 10)[0][0]
            if collapsed_bits not in ideal_bits:
                break

        count = np.append(count, i)
    
    
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)

    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bin_num = int(samples/20) + 5

    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    print('... Number of bins:', len(bins)-1, '...')

    
    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the logical T1 of your steane code over many runs        
def steane_sample_t1(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):        
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 10)[0]
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(10):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    t1_times = np.array([])
    for k in range(samples):
        # initialize our logical state
        rho = realistic_steane(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            count = np.append(count, i)
            # apply circuit
            rho = realistic_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # measure the probability of being in the Logical |1> state from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3))))))))
        
            # probability of being in the 1 logical state
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))
            
            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        circuit_t1 = ((circuit_runs * 29 + 2) * tg)*1e6
        t1_times = np.append(t1_times, circuit_t1)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bins = int(samples/20) + 5

    n, bins, patches = plt.hist(count, bins = bins, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 

    # - - - Fitting a curve to our plot - - - #

    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

    
### - - - - - - Fault Tolerant Steane Code simulation functions - - - - - - ###

### Run the Steane code simulation realistically with paramters and a certain number of iterations.       
def ft_steane_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    # to save time we will just calculate the normal steane and add 2 ancillas
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_state = np.kron(ideal_state, np.kron(zero, zero))
    ideal_state = ancilla_reset(ideal_state, 5)
    
    qubit_error_probs = np.array([])
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(12):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    # add 2 ancillas for the ft version of steane
    initial_state = np.kron(initial_state, np.kron(zero, zero))
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    rho = initial_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    # all_pops = np.array([])
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = realistic_ft_steane(
        rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        # Measurement operator to see if you are in the 0 logical state
        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5))))))))
        
        # probability of being in the 0 logical state
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))
        
        # Measurement operator to see if you are in the 1 logical state
        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5))))))))
        
        # probability of being in the 1 logical state
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))
        
        # any other probability
        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)
    
        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    
    print('- - -')
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 12)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = 'Logical |0>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = 'Logical |1>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running fault tolerant Steane code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - - - -')
    circuit_runs = 1/popt[1]
    if tg!=None:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
        print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
    else:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the state of logical failure for the Steane code
def ft_steane_sample_failure(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on calculating the probability of state measurements overtime...')

    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_ft_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 10)[0]
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(12):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        # Initialize our logical state depending on parameters
        rho = realistic_ft_steane(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = realistic_ft_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # Check if we are still in our ideal state
            collapsed_bits = vector_state_to_bit_state(collapse_dm(rho), 12)[0][0]
            if collapsed_bits not in ideal_bits:
                break

        count = np.append(count, i)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)

    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 12)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bin_num = int(samples/20) + 5

    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    print('... Number of bins:', len(bins)-1, '...')
    
    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the logical T1 of your steane code over many runs        
def ft_steane_sample_t1(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):        
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_ft_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 10)[0]
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(12):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    t1_times = np.array([])
    for k in range(samples):
        # initialize our logical state
        rho = realistic_ft_steane(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            count = np.append(count, i)
            # apply circuit
            rho = realistic_ft_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # measure the probability of being in the Logical |1> state from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5))))))))
        
            # probability of being in the 1 logical state
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))
            
            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        circuit_t1 = ((circuit_runs * 29 + 2) * tg)*1e6
        t1_times = np.append(t1_times, circuit_t1)
        
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 12)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bins = int(samples/20) + 5

    n, bins, patches = plt.hist(count, bins = bins, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 

    # - - - Fitting a curve to our plot - - - #

    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    
### - - - - - - 9-qubit Code simulation functions - - - - - - ###

### Run the nine qubit code simulation realistically with paramters and a certain number of iterations.       
def nine_qubit_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(zero, np.kron(zero, np.kron(
        zero, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
            zero, np.kron(zero, np.kron(zero, zero))))))))))
    
    ideal_state = initialize_nine_qubit_logical_state(initial_state)
    
    qubit_error_probs = np.array([])
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(11):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    rho = initial_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    # all_pops = np.array([])
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = realistic_nine_qubit(
        rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        # Measurement operator to see if you are in the 0 logical state
        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5))))))))
        
        # probability of being in the 0 logical state
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))
        
        # Measurement operator to see if you are in the 1 logical state
        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5))))))))
        
        # probability of being in the 1 logical state
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))
        
        # any other probability
        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)

        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')        
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 11)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = 'Logical |0>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = 'Logical |1>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running fault tolerant Steane code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - - - -')
    circuit_runs = 1/popt[1]
    if tg!=None:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
        print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
    else:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the state of logical failure for the Steane code
def nine_qubit_sample_failure(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on calculating the probability of state measurements overtime...')

    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_ft_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 11)[0]
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(11):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        # Initialize our logical state depending on parameters
        rho = realistic_ft_steane(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = realistic_ft_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # Check if we are still in our ideal state
            collapsed_bits = vector_state_to_bit_state(collapse_dm(rho), 12)[0][0]
            if collapsed_bits not in ideal_bits:
                break

        count = np.append(count, i)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')    
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 11)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bin_num = int(samples/20) + 5

    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    print('... Number of bins:', len(bins)-1, '...')
    
    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Create a plot that samples the logical T1 of your steane code over many runs        
def nine_qubit_sample_t1(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):        
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    ideal_state = initialize_ft_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 11)[0]
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(11):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    t1_times = np.array([])
    for k in range(samples):
        # initialize our logical state
        rho = realistic_ft_steane(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            count = np.append(count, i)
            # apply circuit
            rho = realistic_ft_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

            # measure the probability of being in the Logical |1> state from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5))))))))
        
            # probability of being in the 1 logical state
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))
            
            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        circuit_t1 = ((circuit_runs * 29 + 2) * tg)*1e6
        t1_times = np.append(t1_times, circuit_t1)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')    
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 11)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bins = int(samples/20) + 5

    n, bins, patches = plt.hist(count, bins = bins, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 

    # - - - Fitting a curve to our plot - - - #

    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    