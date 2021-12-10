import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Set parameters
alpha = 0.3
delta = 0.1
A = 1.0
sigma = 1.5
beta = 0.8
b = 0.501
v = 1.554
l = 1
# assume that this person will live for 50 years
S_set = 50
chi = np.ones(S_set)

# Make initial guesses
r_guess = 0.1
b_guess = [0.01]*(S_set-1)
n_guess = [0.2]*S_set
bn_guess = b_guess + n_guess

r, bn, success, euler_errors = SS.SS_solver(r_guess, bn_guess,alpha, delta, A,
                                         sigma, chi, l, v, b, beta)

C_s = (1 + r) *np.array(b_s) + w * np.array(bn_guess[-S_set:]) - np.array(b_sp1)

# plot savings
plt.figure()
plt.plot(range(S_set+1), B_s)
plt.xlabel('Age s')
plt.ylabel('Savings b')
plt.title('Steady-state distribution of saving b')
plt.show()

# plot labor supply
plt.figure()
plt.plot(range(S_set+1), N_s)
plt.xlabel('Age s')
plt.ylabel('Labor Supply')
plt.title('Steady-state distribution of labor supply n')
plt.show()

# plot consumption
plt. figure()
plt.plot(range(S_set), C_s)
plt.xlabel('Age s')
plt.ylabel('Unit of Comsumption')
plt.title('Steady-state distribution of consumption c')
plt.show()
