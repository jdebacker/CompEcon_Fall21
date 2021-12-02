import necessary_equations as ne
import scipy.optimize as opt

# then, let's write a function for the SS algorithm
    # guess r -> w
    # solve for b2 and b3 from hh_foc
    # use MC with b2, b3, n -> K, L
    # use K, L in get_r -> r'
    # check if r' = guess of r
    # loop again if not


def SS_solver(r_guess, b_guesses, n, alpha, delta, A, sigma, beta):
    '''
    Solves for the SS of the economy
    '''
    xi = 0.8
    tol = 1e-8
    max_iter = 500
    dist = 7
    iter = 0
    r = r_guess
    b_sp1 = b_guesses
    while (dist > tol) & (iter < max_iter):
        w = ne.get_w(r, (alpha, delta, A))
        sol = opt.root(
            ne.hh_foc, b_sp1,
            args=(r, w, n, (sigma, beta)))
        b_sp1 = sol.x
        euler_errors = sol.fun
        K = ne.get_K(b_sp1)
        L = ne.get_L(n)
        r_prime = ne.get_r(K, L, (alpha, delta, A))
        dist = (r - r_prime) ** 2
        iter += 1
        r = xi * r + (1 - xi) * r_prime
    success = iter < max_iter

    return r, success, euler_errors

