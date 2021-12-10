import numpy as np

def get_L(n):
    '''
    Function to compute aggregate
    labor supplied
    '''
    L = n.sum()
    return L


def get_K(b):
    '''
    Function to compute aggregate
    capital supplied
    '''
    K = b.sum()
    return K

def get_r(K, L, params):
    '''
    Compute the interest rate from
    the firm's FOC
    '''
    alpha, delta, A = params

    r = alpha * A * (L / K) ** (1 - alpha) - delta
    return r


def get_w(r, params):
    '''
    Solve for the w that is consistent
    with r from the firm's FOC
    '''
    alpha, delta, A = params
    w = (1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha / (1 - alpha))
    return w


def euler_equation(bn_guess, w, r, params):
    sigma, chi, l, v, b, beta = params

    length = len(bn_guess)
    S_set = int((length + 1)/2)

    # assume the individuals are born with no savings and save no income in the last period
    b_s = [0] + list(bn_guess[:S_set-1]) + [0]

    # assume the labor supply of final period is 0
    n_s = list(bn_guess[-S_set:]) + [0]

    euler_error = np.zeros(2*S_set-1)

    for i in range(S_set):
        c = (1 + r) * b_s[i] + w * n_s[i] - b_s[i+1]
        g_n = 1-(n_s[i])**v
        euler_error[i] = w*(c**-sigma) - chi[i] * (b/l) * (n_s[i]/l)**(v-1) * g_n**((1-v)/v)

    for j in range(S_set-1):
        c_s = (1 + r) * b_s[j] + w * n_s[j] - b_s[j+1]
        c_sp1 = (1 + r) * b_s[j+1] + w * n_s[j+1] - b_s[j+2]

        euler_error[S_set+j] = c_s**-sigma - beta * (1+r) * c_sp1**-sigma

    return euler_error
