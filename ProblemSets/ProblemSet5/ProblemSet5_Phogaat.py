#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing packages

import pandas as pd
import numpy as np

import scipy.optimize as opt
import scipy.stats as stats

from geopy.distance import geodesic
from itertools import combinations


# In[2]:


# Read the csv files
df = pd.read_csv('https://raw.githubusercontent.com/jdebacker/CompEcon_Fall21/main/Matching/radio_merger_data.csv')

df.head()


# ### Data Prep 1

# In[3]:


def data_prep(df):
    # Creating Coordinates
    df['buyer_loc']  = df.apply(lambda x : [x['buyer_lat'],x['buyer_long']],axis = 1)
    df['target_loc'] = df.apply(lambda x : [x['target_lat'],x['target_long']],axis = 1)
    df['population_target'] = np.log(df['population_target']/1000)
    
    return (df)

df = data_prep(df)


# ### Data Prep 2

# In[4]:


# Taking all possible combination of each buyer and target
def data_prep2(df):
    
    
    final_data = pd.DataFrame()
    n_comb = list(combinations(range(1,df.shape[0]+1),2))
    
    for i,j in n_comb:
        '''
        First for i variable
        '''
        x1bim = df[df['buyer_id']==i]['num_stations_buyer'].values[0]
        x2bim = df[df['buyer_id']==i]['corp_owner_buyer'].values[0]
        y1tim = df[df['buyer_id']==i]['population_target'].values[0]
        
        buy_iloc = df[df['buyer_id']==i]['buyer_loc'].values[0]
        targ_iloc = df[df['buyer_id']==i]['target_loc'].values[0]
        
        '''
        Now for j variable
        '''
        x1bjm = df[df['buyer_id']==j]['num_stations_buyer'].values[0]
        x2bjm = df[df['buyer_id']==j]['corp_owner_buyer'].values[0]
        y1tjm = df[df['buyer_id']==j]['population_target'].values[0]
        
        buy_jloc = df[df['buyer_id']==j]['buyer_loc'].values[0]
        targ_jloc = df[df['buyer_id']==j]['target_loc'].values[0]
        
        '''
        Distance for all possible combinations
        '''
        dist_biti = geodesic(buy_iloc, targ_iloc).miles
        dist_bjtj = geodesic(buy_jloc, targ_jloc).miles
        dist_bitj = geodesic(buy_iloc, targ_jloc).miles
        dist_bjti = geodesic(buy_jloc, targ_iloc).miles
        
        final_data = final_data.append(pd.DataFrame({'x1bim':[x1bim], 'x2bim':x2bim, 'y1tim':y1tim, 'x1bjm':x1bjm,
                                                    'x2bjm':x2bjm, 'y1tjm':y1tjm, 'dist_biti':dist_biti, 'dist_bjtj':dist_bjtj,
                                                    'dist_bitj':dist_bitj, 'dist_bjti':dist_bjti}))
        
    return(final_data)
        

# Separating for each year
opt_data = []
for year,data in df.groupby('year'):
    print(year)
    final_data = data_prep2(data)
    opt_data.append(final_data)
    

opt_data = pd.concat(opt_data).reset_index(drop = True)


# ### Model Without Transfer

# In[16]:


def opt_func(params,df):
    '''
    The two parameteres: a1 and b1
    '''
    a1,b1 = params

    #Observed payoff components
    df['payoff_biti'] = df['x1bim']*df['y1tim'] + a1*df['x2bim']*df['y1tim'] + b1*df['dist_biti']
    df['payoff_bjtj'] = df['x1bjm']*df['y1tjm'] + a1*df['x2bjm']*df['y1tjm'] + b1*df['dist_bjtj']
    
    #Counterfactual payoff components
    df['payoff_bitj'] = df['x1bim']*df['y1tjm'] + a1*df['x2bim']*df['y1tjm'] + b1*df['dist_bitj']
    df['payoff_bjti'] = df['x1bjm']*df['y1tim'] + a1*df['x2bjm']*df['y1tim'] + b1*df['dist_bjti']
    
    
    #Payoff Calculation
    
    df['observed'] = df['payoff_biti'] + df['payoff_bjtj']
    df['counterfactual'] = df['payoff_bitj'] + df['payoff_bjti']
    
    # Maximize the observations where onserve payoff is more than counterfactual payoff
    return -(df[df['observed']>df['counterfactual']].shape[0])
    
    
# Initial value setting and optimizing model   
params =  (0.2, 0.5)
result = opt.minimize(opt_func,params,args = (opt_data,),
                   method = 'Nelder-Mead')


opt_params = result.x
opt_alpha = opt_params[0]
opt_beta = opt_params[1]


max_score = opt_func(opt_params, opt_data)

# Print the result

# Inversing the sign of maximum score as we have used the function to find the coefficients 
# which gives us the minimum score

print('Model1 Without Transfer:')
print('Optimal value of alpha :', opt_alpha)
print('Optimal value of beta :', opt_beta)
print('Maximum socre:', -(max_score))


# ### Model With Transfer (the prices pay to acquire the target Station)

# In[8]:


def data_prep_hhh(df):
    
    final_data = pd.DataFrame()
    n_comb = list(combinations(range(1,df.shape[0]+1),2))
    
    for i,j in n_comb:
        
        hhhim = df[df['target_id']==i]['hhi_target'].values[0]
        hhhjm = df[df['target_id']==j]['hhi_target'].values[0]
        
        
        
        
        final_data = final_data.append(pd.DataFrame({'hhhim':[hhhim], 'hhhjm':hhhjm}))
        
    return(final_data)
        
        
        

opt_data_hhh = []
for year,data in df.groupby('year'):
    print(year)
    final_data = data_prep_hhh(data)
    opt_data_hhh.append(final_data)
    
# Concatenating HHI dataset with the previous data
opt_data_hhh = pd.concat(opt_data_hhh).reset_index(drop = True)

opt_data_hhh = pd.concat([opt_data,opt_data_hhh],axis = 1)


# In[15]:


def opt_func_hhh(params,df):
    
    a1,b1,d1,g1 = params

    #Observed payvalue off components
    df['payvalue off_biti'] = d1*df['x1bim']*df['y1tim'] + a1*df['x2bim']*df['y1tim'] + b1*df['dist_biti'] + g1*df['hhhim']
    df['payvalue off_bjtj'] = d1*df['x1bjm']*df['y1tjm'] + a1*df['x2bjm']*df['y1tjm'] + b1*df['dist_bjtj'] + g1*df['hhhjm']
    
    #counterfactual payvalue off components
    df['payvalue off_bitj'] = d1*df['x1bim']*df['y1tjm'] + a1*df['x2bim']*df['y1tjm'] + b1*df['dist_bitj'] + g1*df['hhhjm']
    df['payvalue off_bjti'] = d1*df['x1bjm']*df['y1tim'] + a1*df['x2bjm']*df['y1tim'] + b1*df['dist_bjti'] + g1*df['hhhim']
    
    
    #Payvalue off Calculation
    
    df['observed'] = df['payvalue off_biti'] + df['payvalue off_bjtj']
    df['counterfactual'] = df['payvalue off_bitj'] + df['payvalue off_bjti']
    
    return -(df[df['observed']>df['counterfactual']].shape[0])
    
    
    
    

# initial params
params =  (-0.2, -0.1,0.1,0.2)
result_hhh = opt.minimize(opt_func_hhh,params,args = (opt_data_hhh,),
                   method = 'Nelder-Mead')


opt_params = result_hhh.x
opt_alpha = opt_params[0]
opt_beta = opt_params[1]
opt_delta = opt_params[2]
opt_gamma = opt_params[3]

max_score_hhh = opt_func_hhh(opt_params, opt_data_hhh)

# Print the result

# Inversing the sign of maximum score as we have used the function to find the coefficients 
# which gives us the minimum score

print('Model2 With Transfer')
print('Optimal value of delta:', opt_delta)
print('Optimal value of alpha:', opt_alpha)
print('Optimal value of gamma:', opt_gamma)
print('Optimal value of beta:', opt_beta)
print('Maximum score:', -(max_score_hhh))


# In[ ]:




