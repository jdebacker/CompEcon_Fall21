import numpy as np

from functions import bellman_operator
from functions import profit
import matplotlib.pyplot as plt

# set the parameter
beta = 0.6
inv = 0.15
R = 0.1

'''
------------------------------------------------------------------------
Create Grid for State Space    
------------------------------------------------------------------------
lb_w      = scalar, lower bound of cake grid
ub_w      = scalar, upper bound of cake grid 
size_w    = integer, number of grid points in cake state space
w_grid    = vector, size_w x 1 vector of cake grid points 
------------------------------------------------------------------------
'''
lb_w = 2 
ub_w = 100 
size_w = 200  # Number of grid points
w_grid = np.linspace(lb_w, ub_w, size_w)


'''
------------------------------------------------------------------------
Value Function Iteration    
------------------------------------------------------------------------
VFtol     = scalar, tolerance required for value function to converge
VFdist    = scalar, distance between last two value functions
VFmaxiter = integer, maximum number of iterations for value function
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of w and w'
Vstore    = matrix, stores V at each iteration 
VFiter    = integer, current iteration number
V_params  = tuple, contains parameters to pass into Belman operator: beta, sigma
TV        = vector, the value function after applying the Bellman operator
PF        = vector, indicies of choices of w' for all w 
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''

VFtol = 1e-5
VFdist = 7.0 
VFmaxiter = 5000 
V = np.zeros(size_w) #true_VF # initial guess at value function
Vstore = np.zeros((size_w, VFmaxiter)) #initialize Vstore array
VFiter = 1 
V_params = (beta, inv, R)

while VFdist > VFtol and VFiter < VFmaxiter:
    Vstore[:, VFiter] = V
    TV, optW = bellman_operator(V, w_grid, V_params)
    VFdist = (np.absolute(V - TV)).max()  # check distance
    
    V = TV
    VFiter += 1           

VF = V

# Plot value function 
plt.figure()
plt.plot(w_grid[1:], VF[1:])
plt.xlabel('Amount of Oil')
plt.ylabel('Value Function')
plt.title('Value Function - deterministic oil sales')
plt.show()

# Plot value function at several iterations
plt.figure()
fig, ax = plt.subplots()
ax.plot(w_grid, Vstore[:,0], label='1st iter')
ax.plot(w_grid, Vstore[:,2], label='2nd iter')
ax.plot(w_grid, Vstore[:,3], label='3rd iter')
ax.plot(w_grid, Vstore[:,5], label='5th iter')
ax.plot(w_grid, Vstore[:,10], label='10th iter')
ax.plot(w_grid, Vstore[:,VFiter-1], 'k', label='Last iter')
# Now add the legend with some customizations.
legend = ax.legend(loc='lower right', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Amount of Oil')
plt.ylabel('Value Function')
plt.title('Value Function - deterministic oil sales')
plt.show()

#Plot optimal consumption rule as a function of cake size
plt.figure()
fig, ax = plt.subplots()
ax.plot(w_grid[1:], optS[1:], label='Sales')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Amount of Oil')
plt.ylabel('Optimal Sales')
plt.title('Policy Function, sales - deterministic oil sales')
plt.show()

#Plot cake to leave rule as a function of cake size
plt.figure()
fig, ax = plt.subplots()
ax.plot(w_grid[1:], optW[1:], label='Storage')
ax.plot(w_grid[1:], w_grid[1:], '--', label='45 degree line')
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Amount of Oil')
plt.ylabel('Optimal Storage')
plt.title('Policy Function, storage - deterministic oil sales')
plt.show()