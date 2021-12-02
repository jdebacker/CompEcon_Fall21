import SS
import numpy as np


# Set parameters
n = np.array([0.3, 0.5, 0.2])
alpha = 0.3
delta = 0.1
A = 1.0
sigma = 1.5
beta = 0.8

# Make initial guesses
r_guess = 0.1
b_guesses = np.array([0.01, 0.01])

r_ss, success, euler_errors = SS.SS_solver(
    r_guess, b_guesses, n, alpha, delta, A, sigma, beta)

print('The SS interest is ', r_ss, 'Did we find the solution? ', success)
print('The Euler errors are ', euler_errors)
