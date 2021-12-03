# Import packages
import numpy as np

from scipy import interpolate
from scipy.optimize import fminbound

def profit(w, wprime, inv, R):
    
    '''
    w: amount of oil in current period
    wprime: amount of oil saved for next period
    price: price of oil
    i: return rate
    q: amount of newly produced oil
    '''
    
    S = w - wprime/(R+1)
    price = 100/np.exp(S)
    Pi = S * price + inv * wprime
    
    return Pi



def bellman_operator(V, w_grid, params):
    beta, inv, R = params
    
    # Apply cubic interpolation to V
    V_func = interpolate.interp1d(w_grid, V, kind='cubic', fill_value='extrapolate')
    
    # Initialize array for operator and policy function
    TV = np.empty_like(V)
    optW = np.empty_like(TV)
    
    for i, w in enumerate(w_grid):
        
        def objective(wprime):
            return - profit(w, wprime, inv, R) - beta * V_func(wprime)
        wprime_star = fminbound(objective, 1e-6, w - 1e-6)
        optW[i] = wprime_star
        TV[i] = - objective(wprime_star)
    return TV, optW
