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


def mu_c_func(c, sigma):
    '''
    Marginal utility of consumption
    '''
    mu_c = c ** -sigma
    return mu_c


def get_c(r, w, b_s, b_sp1, n):
    '''
    Find consumption using the budget constraint
    and the choice of savings (b_sp1)
    '''
    c = (1 + r) * b_s + w * n - b_sp1
    return c


# solve for b2 and b3, given r and w, from hh_foc
def hh_foc(b_list, r, w, n, params):
    '''
    Define the household first order conditions
    '''
    sigma, beta = params
    b2, b3 = b_list[0], b_list[1]
    b_s = np.array([0.0, b2, b3])
    b_sp1 = np.array([b2, b3, 0.0])
    c = get_c(r, w, b_s, b_sp1, n)
    mu_c = mu_c_func(c, sigma)
    euler_error = mu_c[:-1] - beta * (1 + r) * mu_c[1:]
    # note that euler_error is length 2
    return euler_error
