import necessary_equations as ne
import scipy.optimize as opt


def SS_solver(r_guess, bn_guess, alpha, delta, A, sigma, chi, l, v, b, beta):
    '''
    Solves for the SS of the economy
    '''
    xi = 0.2
    tol = 1e-8
    max_iter = 500
    dist = 7
    iter = 0
    r = r_guess
    bn = bn_guess

    while (dist > tol) & (iter < max_iter):
        w = ne.get_w(r, (alpha, delta, A))

        sol = opt.root(
            ne.euler_equation, bn,
            args=(r, w, (sigma, chi, l, v, b, beta)))

        bn = sol.x
        euler_errors = sol.fun

        b_s = bn[:S_set-1]
        n_s = bn[-S_set:]

        K = ne.get_K(b_s)
        L = ne.get_L(n_s)
        r_prime = ne.get_r(K, L, (alpha, delta, A))
        dist = (r - r_prime) ** 2
        iter += 1
        r = xi * r_prime + (1 - xi) * r


    success = iter < max_iter

    return r, bn, success, euler_errors
