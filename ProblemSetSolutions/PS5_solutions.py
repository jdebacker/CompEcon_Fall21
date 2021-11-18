#Import the packages we would use
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import pandas as pd
from geopy.distance import distance
import os
from scipy.optimize import differential_evolution

#Importing the data file
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'Matching', 'radio_merger_data.csv')
print(data_dir)

#Importing the data file (This section works)
filepath= os.path.join("..", "Matching","radio_merger_data.csv")
df = pd.read_csv(filepath)

df07 = df.loc[df['year'] == 2007]
df08 = df.loc[df['year'] == 2008]
years = [df07, df08]
'''
We need actual and counterfactual data.
Buy = ['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'num_stations_buyer', 'corp_owner_buyer']
Target = ['target_id', 'target_lat', 'target_long', 'scaled_price', 'hhi_target', 'scaled_pop']

counterfac = [x[Buy].iloc[i].values.tolist() + x[Target].iloc[j].values.tolist()
             for x in years for i in range(len(x) - 1)
             for j in range(i + 1, len(x))]
counterfactuals = pd.DataFrame(counterfac, columns = Buy + Target)

We will use the longtitude and latitude to calculate the distance.
counterfactuals['buyer_loc'] = counterfactuals[['buyer_lat', 'buyer_long']].apply(tuple, axis=1)
counterfactuals['target_loc'] = counterfactuals[['target_lat', 'target_long']].apply(tuple, axis=1)
counterfactuals['distance_mi'] = counterfactuals.apply(lambda row: distance(row['buyer_loc'], row['target_loc']).miles, axis=1)
'''

################################################################
#Codes in response to "PS5 solutions: make arrays of matches"
def create_array_ids(x):
    '''
    Args
    ----------
    x : df07array for the year 2007
        df08array for the year 2008 (only put the array for a year)

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
# code for condensed create_vars function that we can call 4 times 

def create_x(merger_df, id_array):
    '''
    Args:
        merger_df: original dataframe with matches and match characteristics
        id_array: array with all matches and counterfactuals for years 2007 and 2008; 
            will specify which pairs needed (either real or counterfactual)
    Returns: 
        dframe: dataframe of specified buyer and target pairs with characteristics and calculated variables
    '''
    #create additional variables for the X matrix inresponse to the question
    merger_df['scaled_pop'] = merger_df['population_target']/1000000
    merger_df['scaled_price'] = merger_df['price']/1000000
    merger_df['buyer_loc'] = merger_df[['buyer_lat', 'buyer_long']].apply(tuple, axis=1)
    merger_df['target_loc'] = merger_df[['target_lat', 'target_long']].apply(tuple, axis=1)
    
    #turn id_array into a dataframe
    id=pd.DataFrame(id_array, columns=['year', 'buyer_id', 'target_id'])
    
    #create buyer columns including locations and number of stations
    i=merger_df[['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'num_stations_buyer', 'corp_owner_buyer', 'buyer_loc']]
    #create seller columns including locations, price, and hhi
    j=merger_df[['year', 'target_id', 'target_lat', 'target_long', 'scaled_price', 'hhi_target', 'scaled_pop', 'target_loc']]

    #merge buyer and target characteristics from dataframe onto id dataframe by year, buyerid, and targetid 
    df_buyer = pd.merge(id, i, on=['year','buyer_id'], how='left')
    df_target = pd.merge(id, j, on=['year','target_id'], how='left')
    dframe = pd.concat([df_buyer, df_target], axis=1)
    
    #add distance variable
    dframe['distance_mi'] = dframe.apply(lambda row: distance(row['buyer_loc'], row['target_loc']).miles, axis=1)

    return(dframe)

df_1 = create_x(df, array_ids_and_years[:, [0,1,2]])
df_1 = df_1.loc[:,~df_1.columns.duplicated()]
df_1 = df_1.add_suffix('1') #take care of the same variable name problem when we concat

df_2 = create_x(df, array_ids_and_years[:, [0,3,4]])
df_2 = df_2.loc[:,~df_2.columns.duplicated()]
df_2 = df_2.add_suffix('2')

df_3 = create_x(df, array_ids_and_years[:, [0,5,6]])
df_3 = df_3.loc[:,~df_3.columns.duplicated()]
df_3 = df_3.add_suffix('3')

df_4 = create_x(df, array_ids_and_years[:, [0,7,8]])
df_4 = df_4.loc[:,~df_4.columns.duplicated()]
df_4 = df_4.add_suffix('4')
################################################################

################################################################
#payoff without transfers

#first combine the actual and conterfactual data for the payoff function
df_all=pd.concat([df_1,df_2,df_3,df_4], axis=1)
df=pd.concat([df_1,df_2,df_3,df_4], axis=1)

#create the payoff function
def payoff_without_transfers(df, parameters):
    '''
    Args:
        df_all: data used for calculation. The actual and counter factual data must be in teh same row
        parameters: initial parameter estimates for alpha and beta

    Returns:
        f: the payoff to the merger
        
    Model 1:
    f(b; t) = num_stations_buyer_bm * population_target_tm 
              + alpha * corp_owner_buyer_bm * population_target_tm 
              + beta * distancebtm + error_btm
    
    '''
    alpha = parameters[0]
    beta = parameters[1]

    f1 = df['num_stations_buyer1']*df['scaled_pop1'] + alpha * df['corp_owner_buyer1'] * df['scaled_pop1'] + beta * df['distance_mi1']
    f2 = df['num_stations_buyer2']*df['scaled_pop2'] + alpha * df['corp_owner_buyer2'] * df['scaled_pop2'] + beta * df['distance_mi2']
    f3 = df['num_stations_buyer3']*df['scaled_pop3'] + alpha * df['corp_owner_buyer3'] * df['scaled_pop3'] + beta * df['distance_mi3']
    f4 = df['num_stations_buyer4']*df['scaled_pop4'] + alpha * df['corp_owner_buyer4'] * df['scaled_pop4'] + beta * df['distance_mi4']

    #f (b,t) + f(bdot, tdot)  ≥ f(b, tdot) + f(bdot, t)
    score = (f1+f2) > (f3+f4)
    neg_total_score = score.sum()
    return neg_total_score

#Initial Guess
initial_guess = (0.5, 0.5)
a=payoff_without_transfers(df, initial_guess)
parm_estimates = opt.minimize(payoff_without_transfers, initial_guess, args=df, method = 'Nelder-Mead')

set_bound = [(-0.5,0.5),(-0.5,0.5)]
parm_estimates = differential_evolution(payoff_without_transfers, set_bound, args=(df,))
################################################################


#################################### Without transfers

# Define the payoff function (I believe this section works)

def payoff(data, parameters):
    '''
    Args:
        data: actual and counterfactual data for buyers and targets
        parameters: initial parameter estimates for alpha and beta

    Returns:
        f: the payoff to the merger
    '''
    # note: there is no coefficent for the first term
    alpha = parameters[0]
    beta = parameters[1]

    f = data['num_stations_buyer'] * data['scaled_pop'] + alpha * data['corp_owner_buyer'] * data['scaled_pop'] + beta * data['distance_mi']

    return(f)

params = (0.5, 0.5)

# actual payoffs for year 2007 and 2008
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

# Objective Function (this part doesn't work)
def objective(self, actual, counter):
    '''
    A function that returns the maximum score estimator(MSE)

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
# Payoff transfer function (I think this works)

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

# actual payoffs for year 2007
actual07 = pd.DataFrame(payofftransfers(data = df07, parameters = params2))

# actual 2008 payoffs for year 2008
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
