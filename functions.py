# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Create a grid w, using 300 points between 0.1 and 4.
'''
------------------------------------------------------------------------
Create Grid for State Space    
------------------------------------------------------------------------
lb_m      = scalar, lower bound of money grid
ub_m      = scalar, upper bound of money grid 
size_m    = integer, number of grid points in money state space
m_grid    = vector, size_w x 1 vector of money grid points 
------------------------------------------------------------------------
'''
m = np.linspace(0.1,4,300)
m_lb = 0.1
m_ub = 4
size_m = 300
m_grid = np.linspace(m_lb, m_ub, size_m)

# Step 1 Define parameters
sigma = 1.5
rho = 0.05
r = 0.92
R = 1

# Step 2 Create grid of current profit (like utility) values 


'''
------------------------------------------------------------------------
Create grid of current utility values    
------------------------------------------------------------------------
I        = matrix, current investment (I = m-m')
P        = matrix, current period profit value for all possible
           choices of m and m' (rows are m, columns m')
------------------------------------------------------------------------
'''

def p_func(size_m, rho, R, sigma):
    I = np.zeros((size_m, size_m)) 
    for i in range(size_m): # loop over m
        for j in range(size_m): # loop over m'
            I[i, j] = (m_grid[i] - m_grid[j]) * (1 + rho)/R # note that if m'>m, investment is negative
    # replace 0 and negative consumption with a tiny value 
    # This is a way to impose non-negativity on cons
    I[I<=0] = 1e-15
    if sigma == 1:
        P = np.log(I)
    else:
        P = (I ** (1 - sigma)) / (1 - sigma) # because the profit function has negative second order direvitive, here we use similar function like the cake example
    P[I<0] = -9999999

    return P

# Step 3 Build up VFI algorithm and value difference function
'''
------------------------------------------------------------------------
Value Function Iteration    
------------------------------------------------------------------------
VFtol     = scalar, tolerance required for value function to converge
VFdist    = scalar, distance between last two value functions
VFmaxiter = integer, maximum number of iterations for value function
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of m and m'
Vstore    = matrix, stores V at each iteration 
VFiter    = integer, current iteration number
TV        = vector, the value function after applying the Bellman operator
PF        = vector, indicies of choices of m' for all m 
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''

def v_func(size_m, r, P ):
    VFtol = 0.0001
    VFdist = 7.0 
    VFmaxiter = 3000 
    V = np.zeros(size_m) # initial guess at value function
    Vmat = np.zeros((size_m, size_m)) # initialize Vmat matrix
    Vstore = np.zeros((size_m, VFmaxiter)) # initialize Vstore array
    VFiter = 1 
    while VFdist > VFtol and VFiter < VFmaxiter:  
        for i in range(size_m): # loop over m
            for j in range(size_m): # loop over m'
                Vmat[i, j] = P[i, j] + r * V[j] # p((1+ρ-r) / [(1+ρ) * M1 +M2]) + p (M2)
        
        Vstore[:, VFiter] = V.reshape(size_m,) # store value function at each iteration for graphing later
        TV = Vmat.max(1) # apply max operator to Vmat (to get V(m))
        PF = np.argmax(Vmat, axis=1)
        VFdist = (np.absolute(V - TV)).max()  # check distance
        V = TV
        VFiter += 1 
        


    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')


    VF = V # solution to the functional equation
    return VF, PF

# Step 4 Extract decision rules from solution

'''
------------------------------------------------------------------------
Find investment and remaining money policy functions   
------------------------------------------------------------------------
optM  = vector, the optimal choice of m' for each m
optI  = vector, the optimal choice of i for each i
------------------------------------------------------------------------
'''

def gra(VF, PF):
    optM = m_grid[PF] # tomorrow's optimal remaining money amount (savings function)
    optI = m_grid - optM # optimal investment - get investment through the transition equation

    # Visualize output
    # Plot value function 
    plt.figure()
    plt.scatter(m_grid[1:], VF[1:])
    plt.xlabel('Money Amount')
    plt.ylabel('Value Function')
    plt.title('Value Function - deterministic investment choice')
    plt.show()

    #Plot optimal investment rule as a function of money amount (state variable)
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(m_grid[3:], optI[3:], label='Investment')

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper left', shadow=False)

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.xlabel('Money Amount')
    plt.ylabel('Optimal Investment')
    plt.title('Policy Function, investment - deterministic money spending')
    plt.show()
