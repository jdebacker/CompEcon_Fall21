import PS8code as P8
import numpy as np
# Define the parameters
'''
------------------------------------------------------------------------
Define the parameters
------------------------------------------------------------------------
lb_m      = scalar, lower bound of money grid
ub_m      = scalar, upper bound of money grid 
size_m    = integer, number of grid points in money state space
m_grid    = vector, size_w x 1 vector of money grid points 
------------------------------------------------------------------------
I         = matrix, current investment (I = m-m')
P         = matrix, current period profit value for all possible
           choices of m and m' (rows are m, columns m')
------------------------------------------------------------------------
sigma     = the parameter for the profit function
rho       = the interest of the ramianing money
r         = the gaining from investment
'''
m = np.linspace(0.1,4,300)
m_lb = 0.1
m_ub = 4
size_m = 300
m_grid = np.linspace(m_lb, m_ub, size_m)

sigma = 1.5
rho = 0.05
r = 0.92
R = 1

# Call the main functions
P = P8.p_func(size_m, rho, R, sigma)
VF, PF = P8.v_func(size_m, r, P )

print('solution to the functional equation', VF)

# Show the graph
P8.gra(VF, PF)