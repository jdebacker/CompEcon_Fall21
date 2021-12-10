# mport the packages
import pandas as pd
import numpy as np

import scipy.optimize as opt
import scipy.stats as stats

from geopy.distance import geodesic

'''
Part1 Data Process
'''
# read in data
df = pd.read_csv('radio_merger_data.csv')

# combine latitude and longitude into a tuple for distance calculation
buyer_loc = df[['buyer_lat','buyer_long']]
df['buyer_loc'] = buyer_loc.apply(tuple,axis=1)

target_loc = df[['target_lat','target_long']]
df['target_loc'] = target_loc.apply(tuple,axis=1)

# rescale the population: put population into logs of thousands of people
df['population_target'] = np.log(df['population_target']/1000)

'''
Model 1

Part2 Create a new dataset in which every buyer and target in each market can match with each other.
Also, I only keep the data needed for the model
'''

# create empty lists to hold all the corresponding values
x_1bim = []
y_1tim = []
x_2bim = []
distance_iim = []

x_1bjm = []
y_1tjm = []
x_2bjm = []
distance_jjm = []

distance_ijm = []
distance_jim = []

# seperate the market for different years
mkt = df['year'].unique()
no_mkt = len(mkt)

# first loop for different year/market
for m in range(no_mkt):
    data = df[df['year'].isin([mkt[m]])] # this dataframe only holds data for corresponding market/year

    n,_ = data.shape # get the number of buyers/targets in this market

    for i in range(n-1):
        for j in range(i,n-1):
            j += 1 # this netted loop allows me to get all the observed and counterfactual matches

            # put all the corresponding valurs into the lists
            x_1bim.append(data.loc[(data.buyer_id==i+1),'num_stations_buyer'].values[0])
            y_1tim.append(data.loc[(data.target_id==i+1),'population_target'].values[0])
            x_2bim.append(data.loc[(data.buyer_id==i+1),'corp_owner_buyer'].values[0])

            bi_loc = data.loc[(data.buyer_id==i+1),'buyer_loc'].values[0]
            ti_loc = data.loc[(data.target_id==i+1),'target_loc'].values[0]
            distance_iim.append(geodesic(bi_loc, ti_loc).miles) # calculate the distance btw the buyer and target

            x_1bjm.append(data.loc[(data.buyer_id==j+1),'num_stations_buyer'].values[0])
            y_1tjm.append(data.loc[(data.target_id==j+1),'population_target'].values[0])
            x_2bjm.append(data.loc[(data.buyer_id==j+1),'corp_owner_buyer'].values[0])

            bj_loc = data.loc[(data.buyer_id==j+1),'buyer_loc'].values[0]
            tj_loc = data.loc[(data.target_id==j+1),'target_loc'].values[0]

            distance_jjm.append(geodesic(bj_loc, tj_loc).miles)

            distance_ijm.append(geodesic(bi_loc, tj_loc).miles)
            distance_jim.append(geodesic(bj_loc, ti_loc).miles)

# give these lists a suitable names
# I use the variable names in the model
data_match = {'x_1bim':x_1bim,
              'y_1tim':y_1tim,
              'x_2bim':x_2bim,
              'distance_iim':distance_iim,
              'x_1bjm': x_1bjm,
              'y_1tjm': y_1tjm,
              'x_2bjm': x_2bjm,
              'distance_jjm': distance_jjm,
              'distance_ijm': distance_ijm,
              'distance_jim':distance_jim}


# put all the lists into a dataframe called df_match
df_match = pd.DataFrame(data_match)

'''
Part3 Define Model1
'''

def Q(paras,df):
    '''
    The maximum socre objective function.

    Args:
        params (tuple): model parameters
        data (Pandas DataFrame): data, contains covariates in model

    Returns:
        scalar: the negative value of the score function
            We'll use a negative value so we can use a minimizer to max the sum
    '''

    # unpack tuple of parameters
    alpha, beta = paras

    # find value for each variable to calculate payoff for buyer i and target i
    x_1bim = df['x_1bim']
    y_1tim = df['y_1tim']
    x_2bim = df['x_2bim']
    distance_iim = df['distance_iim']

    # calculate payoff for buyer i and target i
    df['f_ii'] = x_1bim*y_1tim + alpha*x_2bim*y_1tim + beta*distance_iim

    # find value for each variable to calculate payoff for buyer j and target j
    x_1bjm = df['x_1bjm']
    y_1tjm = df['y_1tjm']
    x_2bjm = df['x_2bjm']
    distance_jjm = df['distance_jjm']

    # calculate payoff for buyer j and target j
    df['f_jj'] = x_1bjm*y_1tjm + alpha*x_2bjm*y_1tjm + beta*distance_jjm

    distance_ijm = df['distance_ijm']

    # calculate payoff for buyer i and target j
    df['f_ij'] = x_1bim*y_1tjm + alpha*x_2bim*y_1tjm + beta*distance_ijm

    distance_jim = df['distance_jim']

    # calculate payoff for buyer j and target i
    df['f_ji'] = x_1bjm*y_1tim + alpha*x_2bjm*y_1tim + beta*distance_jim

    df['observed'] = df['f_ii'] + df['f_jj'] # calculate payoff for the observed matches
    df['counterfactual'] = df['f_ij'] + df['f_ji']  # calculate payoff for the counterfactual matches

    # create a list where payoff of observed matches is larger than that of counterfactual matches
    correct = (df['observed']>df['counterfactual'])

    return -sum(correct) # return the negative number that satisfy that condition


'''
Part4 Estimate Model1
'''
# set initial guessess
paras =  (0.1, 0.1)

# minimize the maximum socre objective function
res = opt.minimize(Q, paras, args = (df_match,),
                   method = 'Nelder-Mead')

# get the result of the estimated parameters
est_paras = res.x
# get the result of the estimated alpha
est_alpha = est_paras[0]
est_beta = est_paras[1]

# use the estimated parameters to calculate the maximum score
MS1 = Q(est_paras, df_match)

# Print the resukt
print('For the model1:')
print('The estimate of alpha is', est_alpha)
print('The estimate of beta is', est_beta)
print('The maximum socre estimator is', MS1)

'''
Model 2
Part5 Add a new varibale HHI for model2 and put all the data into new dataset df_match2
'''

HHI_im = []
HHI_jm = []

for m in range(no_mkt):
    data = df[df['year'].isin([mkt[m]])]

    n, _ = data.shape
    for i in range(n-1):
        for j in range(i,n-1):
            j += 1
            HHI_im.append(data.loc[(data.target_id==i+1),'hhi_target'].values[0])
            HHI_jm.append(data.loc[(data.target_id==j+1),'hhi_target'].values[0])
df_match2 = df_match
df_match2['HHI_im'] = HHI_im
df_match2['HHI_jm'] = HHI_jm


'''
Part6 Define Model2
'''

def Q2(paras,df):
    '''
    The maximum socre objective function.

    Args:
        params (tuple): model parameters
        data (Pandas DataFrame): data, contains covariates in model

    Returns:
        scalar: the negative value of the score function
            We'll use a negative value so we can use a minimizer to max the sum
    '''

    delta, alpha, gamma, beta, = paras

    x_1bim = df['x_1bim']
    y_1tim = df['y_1tim']
    x_2bim = df['x_2bim']
    HHI_im = df['HHI_im']
    distance_iim = df['distance_iim']


    df['f_ii'] = delta*x_1bim*y_1tim + alpha*x_2bim*y_1tim + gamma*HHI_im + beta*distance_iim

    # calculate f_(j,j)
    x_1bjm = df['x_1bjm']
    y_1tjm = df['y_1tjm']
    x_2bjm = df['x_2bjm']
    HHI_jm = df['HHI_jm']
    distance_jjm = df['distance_jjm']

    df['f_jj'] = delta*x_1bjm*y_1tjm + alpha*x_2bjm*y_1tjm + gamma*HHI_jm + beta*distance_jjm

    # calculate f_(i,j)

    distance_ijm = df['distance_ijm']

    df['f_ij'] = delta*x_1bim*y_1tjm + alpha*x_2bim*y_1tjm + gamma*HHI_jm + beta*distance_ijm

    # calculate f_(j,i)

    distance_jim = df['distance_jim']

    df['f_ji'] = delta*x_1bjm*y_1tim + alpha*x_2bjm*y_1tim + gamma*HHI_im + beta*distance_jim

    df['observed'] = df['f_ii'] + df['f_jj']
    df['counterfactual'] = df['f_ij'] + df['f_ji']

    correct = (df['observed']>df['counterfactual'])

    return -sum(correct)


'''
Part7 Estimate Model2
'''
# initial guesses
paras =  (0.1, 0.1, 0.1, -0.1)

# minimize the maximum socre objective function
res1 = opt.minimize(Q2, paras, args = (df_match2,),
                   method = 'Nelder-Mead')

# use improved parameters to estimate
res2 = opt.minimize(Q2, res1.x, args = (df_match2,),
                   method = 'Nelder-Mead')

# get the result of the estimated parameters
est_paras2 = res2.x
# get the result of the estimated alpha
est_delta2 = est_paras2[0]
est_alpha2 = est_paras2[1]
est_gamma2 = est_paras2[2]
est_beta2 = est_paras2[3]

# use the estimated parameters to calculate the maximum score
MS2 = Q2(est_paras2, df_match2)

# Print the resukt
print('For the model2:')
print('The estimate of delta is', est_delta2)
print('The estimate of alpha is', est_alpha2)
print('The estimate of gamma is', est_gamma2)
print('The estimate of beta is', est_beta2)
print('The maximum socre estimator is', MS2)
