import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import pandas as pd
from geopy.distance import distance
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'Matching', 'radio_merger_data.csv')
print(data_dir)
# This section works
filepath= os.path.join("..", "Matching","radio_merger_data.csv")
df = pd.read_csv(filepath)

df07 = df.loc[df['year'] == 2007]
df08 = df.loc[df['year'] == 2008]
years = [df07, df08]

Buy = ['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'num_stations_buyer', 'corp_owner_buyer']
Target = ['target_id', 'target_lat', 'target_long', 'scaled_price', 'hhi_target', 'scaled_pop']

counterfac = [x[Buy].iloc[i].values.tolist() + x[Target].iloc[j].values.tolist()
             for x in years for i in range(len(x) - 1)
             for j in range(i + 1, len(x))]
counterfactuals = pd.DataFrame(counterfac, columns = Buy + Target)

counterfactuals['buyer_loc'] = counterfactuals[['buyer_lat', 'buyer_long']].apply(tuple, axis=1)
counterfactuals['target_loc'] = counterfactuals[['target_lat', 'target_long']].apply(tuple, axis=1)
counterfactuals['distance_mi'] = counterfactuals.apply(lambda row: distance(row['buyer_loc'], row['target_loc']).miles, axis=1)


################################################################
#Codes in response to "PS5 solutions: make arrays of matches"
def create_array_ids(x):
    '''
    args
    ----------
    x : df07array or df08array (only put the array for a year)

    Returns
    -------
    arrays that include actual and counterfactual year and ids
    actual ids:
        array1: (b, t)
        array2: (b', t')
    counterfactual ids:
        array3: (b, t')
        array4: (b', t)
    '''
    k = x.shape[1] #ncol
    a1=np.zeros((0, k), int) #for array 1
    a2=np.zeros((0, k), int) #for array 2
    a3=np.zeros((0, k), int) #for array 3
    a4=np.zeros((0, k), int) #for array 4
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            aa=[x[:,0][i], x[:,1][i], x[:,2][j]]
            a3 = np.append(a3, np.array([aa]), axis=0)
            bb=[x[:,0][i], x[:,1][j], x[:,2][i]]
            a4 = np.append(a4, np.array([bb]), axis=0)
            cc=[x[:,0][i], x[:,1][i], x[:,2][i]]
            a1 = np.append(a1, np.array([cc]), axis=0)
            dd=[x[:,0][i], x[:,1][j], x[:,2][j]]
            a2 = np.append(a2, np.array([dd]), axis=0)
    all_a = np.concatenate((a1, a2, a3, a4),axis=1)
    all_a = np.delete(all_a,[3,6,9],axis=1)
    return all_a

#set up arries for the function
df07array=np.array(df07[['year', 'buyer_id','target_id']])
df08array=np.array(df08[['year', 'buyer_id','target_id']])

#run the function by year
a07=create_array_ids(df07array)
a08=create_array_ids(df08array)

#array combinations of buyers and targets
array_ids_and_years=np.concatenate((a07,a08),axis=0)

#create a df so you know the column names
column_names = ['year', 'buyer_id_bt', 'target_id_bt', 'buyer_id_bdot_tdot', 'target_id_bdot_tdot',
                'buyer_id_b_tdot', 'target_id_b_tdot','buyer_id_bdot_t', 'target_id_bdot_t']
df_ids_and_years = pd.DataFrame(data = array_ids_and_years,
                  columns = column_names)
################################################################

#######################################################
# code in response to issue #42
id_array_bt07 = a07[:, [0,1,2]]
id_array_b_t_07 = a07[:, [0,3,4]]
id_array_bt_07 = a07[:, [0,5,6]]
id_array_b_t07 = a07[:, [0,7,8]]
id_array_bt08 = a08[:, [0,1,2]]
id_array_b_t_08 = a08[:, [0,3,4]]
id_array_bt_08 = a08[:, [0,5,6]]
id_array_b_t08 = a08[:, [0,7,8]]
def create_vars(df, id_array):
    '''
    docstring
    '''
    df['scaled_pop'] = df['population_target']/1000000
    df['scaled_price'] = df['price']/1000000
    df['buyer_loc'] = df[['buyer_lat', 'buyer_long']].apply(tuple, axis=1)
    df['target_loc'] = df[['target_lat', 'target_long']].apply(tuple, axis=1)
    df['distance_mi'] = df.apply(lambda row: distance(row['buyer_loc'], row['target_loc']).miles, axis=1)
    id_df = pd.DataFrame(id_array, columns = ['year', 'buyer_id', 'target_id'])

    X = pd.merge(id_df, df[['buyer_id', 'target_id', 'num_stations_buyer', 'corp_owner_buyer','scaled_price', 'hhi_target', 'scaled_pop','distance_mi']], on=['buyer_id', 'target_id'], how='left')

    return(X)


print(create_vars(df, id_array_bt_07))

#############################################################

#################################### Without transfers

# I believe this section works

def payoff(data, parameters):
    '''
    Args:
        data: data used for calculation
        parameters: initial parameter estimates for alpha and beta

    Returns:
        f: the payoff to the merger
    '''
    # note no coef on the first term
    alpha = parameters[0]
    beta = parameters[1]

    f = data['num_stations_buyer'] * data['scaled_pop'] + alpha * data['corp_owner_buyer'] * data['scaled_pop'] + beta * data['distance_mi']

    return(f)

params = (0.5, 0.5)

# actual payoffs
actual7 = pd.DataFrame(payoff(data = df07, parameters = params))
actual8 = pd.DataFrame(payoff(data = df08, parameters = params))
actual = pd.DataFrame(payoff(data=df, parameters = params), columns=['payoff'])

# concat df with payoffs
df = pd.concat([df, actual], axis=1)

# counterfactual 2007 payoffs
counter7 = pd.DataFrame(payoff(data=counterfactuals, parameters=params))
counter7 = counter7[counter7.index < 990]

# counterfactual 2008 payoffs
counter8 = pd.DataFrame(payoff(data=counterfactuals, parameters=params))
counter8 = counter8[counter8.index > 989]

# concat counterfactuals with payoffs
counter = pd.DataFrame(payoff(data=counterfactuals, parameters=params), columns=['payoff'])
counterfactuals = pd.concat([counterfactuals, counter], axis=1)

# this part doesn't work
def objective(self, actual, counter):
    '''
    A function that returns the maximum score estimator

    Args:
        actual: df of actual matches, including payoffs
        counter: df of counterfactual matches, including payoffs

    Returns:
        score: maximum score estimator
    '''
    score = 0
    for x in [actual, counter]:
        for i in range(len(actual)):
            for j in range(len(counter)):
                actual_sum = actual.loc[actual['payoff'][i]] + actual.loc[actual['payoff'][i+1]]
                counter_sum = counter.loc[counter['payoff'][j]] + counter.loc[counter['payoff'][j-1]]
    for i in range(len(actual)):
        if actual_sum >= counter_sum:
            score =+ 1
        else:
            score =+0

    return(score)

results = opt.minimize(objective, params, args = (df, counterfactuals), method = 'Nelder-Mead', options = {'maxiter': 5000})
print("Without Transfers Results:", results)

############################# With transfers
# I think this works

def payofftransfers(data, parameters):
    '''
    Args:
        data: data used for calculation
        parameters: initial parameter estimates for delta, alpha, gamma, and beta

    Returns:
        f: the payoff to the merger
    '''
    delta = parameters[0]
    alpha = parameters[1]
    gamma = parameters[2]
    beta = parameters[3]

    f2 = delta * data['num_stations_buyer'] * data['scaled_pop'] + alpha * data['corp_owner_buyer'] * data['scaled_pop'] + gamma * data['hhi_target'] + beta * data['distance_mi']

    return(f2)

params2 = (0.5, 0.5, 0.5, 0.5)

# actual payoffs
actual07 = pd.DataFrame(payofftransfers(data = df07, parameters = params2))

# actual 2008 payoffs
actual08 = pd.DataFrame(payofftransfers(data = df08, parameters = params2))

# counterfactual 2007 payoffs
counter07 = pd.DataFrame(payofftransfers(data=counterfactuals, parameters=params2))
counter07 = counter07[counter07.index < 990]
# counterfactual 2008 payoffs
counter08 = pd.DataFrame(payofftransfers(data=counterfactuals, parameters=paramss))
counter08 = counter8[counter8.index > 989]

# this part doesn't work

# trying a different kind of syntax from the other one to see if it works. It doesn't
def objective2(self, actual07, actual08, counter07, counter08):
    '''
    A function that returns the maximum score estimator

    Args:
        actual07: actual payoffs from 2007
        actual08: actual payoffs from 2008
        counter07: counterfactual payoffs from 2007
        counter08: counterfactual payoffs from 2008

    Returns:
        score: maximum score estimator
    '''
    price07 = df07['scaled_price'].tolist()
    price08 = df08['scaled_price'].tolist()
    score = [1 for x in [[actual07, counter07, price07], [actual08, counter08, price08]]
            for i in range(len(m[0]))
            for j in range(len(m[0]))
            if (x[0][i] - x[1][j, i] >= x[2][i] - x[2][j]) & (x[0][j] - x[1][i, (j - 1)] >= x[2][j] - x[2][i])]

    return(score)

results = opt.minimize(objective2, params2, args = (actual07, actual08, counter07, counter08), method = 'Nelder-Mead', options = {'maxiter': 5000})
print("With Transfers Results:", results)
