import numpy as np
import pandas as pd
from scipy.optimize import minimize
from geopy.distance import geodesic
from scipy.optimize import differential_evolution

df = pd.read_csv("radio_merger_data.csv")

##################################################################
###Reconstruct the data###
# put these into millions of dollars and people
df['price_mil'] = df['price']/1000000
df['population_target_mil'] = df['population_target']/1000000
##################################################################


##################################################################
### Create a function that calculate distance for a year###
def counterfactual_df_of_year(x, year, buyer_colnames, target_colnames):
    '''
    x: dataframe name
    year: year of the data
    buyer_colnames: the names of the columns for the buyer
    target_colnames: the names of the columns for the target
    
    Return: the counterfactual data of a year
    '''
    x= x[(x.year==year)].reset_index()
    a=[]
    for i in range(len(x["buyer_id"])-1):
        for j in range(i+1,len(x["buyer_id"])):
            if i!= j:
                aa=x[buyer_colnames].iloc[i].values.tolist() + x[target_colnames].iloc[j].values.tolist()
                a.append(aa)
    newdf=pd.DataFrame(a, columns = buyer_colnames + target_colnames)
    return newdf
##################################################################


##################################################################
### Create a function that calculate distance for a year###
def distances_of_year(x,year):
    '''
    calculate the distance between the two objects in a year
    x: dataframe name
    year: year of the data
    Return: a dataframe of a year that include the distance variable
    '''
    x=x[(x.year==year)].reset_index()
    for i in range(len(x["buyer_id"])):
        x.loc[i, "distance"] = geodesic((x.loc[i,"buyer_lat"],x.loc[i,"buyer_long"]), 
                                        (x.loc[i,"target_lat"], x.loc[i,"target_long"])).miles
    return x
##################################################################


##################################################################
###Model 1 function###
#def Model1(df07, df_c07, df08, df_c08):
def Model1(df07, df08, df_c07, df_c08):
    def payoff(parms):
        '''
        create a function that calculate the parameter of model 1
        the inputs are the variable names, all must be in df format

        Model 1:
        f(b; t) = num_stations_buyer_bm * population_target_tm 
                   + alpha * corp_owner_buyer_bm * population_target_tm 
                   + beta * distancebtm + error_btm
        f (b,t) + f (bdot, tdot)  ≥ f (bdot, t) + f(b, tdot)
        '''
        alpha, beta = parms
        
        f07 = df07['num_stations_buyer']*df07['population_target'] + alpha * df07['corp_owner_buyer'] * df07['population_target'] + beta *df07['distance']
        f08 = df08['num_stations_buyer']*df08['population_target'] + alpha * df08['corp_owner_buyer'] * df08['population_target'] + beta *df08['distance']
           
        f_count07 = df_c07['num_stations_buyer']*df_c07['population_target'] + alpha * df_c07['corp_owner_buyer'] * df_c07['population_target'] + beta *df_c07['distance']
        f_count08 = df_c08['num_stations_buyer']*df_c08['population_target'] + alpha * df_c08['corp_owner_buyer'] * df_c08['population_target'] + beta *df_c08['distance']

        a07=[]
        for i in len(df07[df07.buyer_id <= 989]):
            for j in len(df07[df07.buyer_id <= 989]):
                if i==j & f07 > f_count07:
                    a= 1
                    a07.append(a)      
        a08=[]
        for i in len(df08[df08.buyer_id >= 989]):
            for j in len(df08[df08.buyer_id >= 989]):
                if i==j & f08 > f_count08:
                    a= 1
                    a08.append(a)        
        
        joinedlist = a07 + a08
        #
        neg_total_score = -joinedlist.sum()
        return neg_total_score
    
    #Initial Guess
    guess = (0.5, 0.5)
    #set_bound = [(-0.5, 0.5),(-0.5, 0.5)]
    #calculate the parameter estimates
    #parm_estimates = differential_evolution(payoff, set_bound)
    parm_estimates = minimize(payoff, guess, method = 'Nelder-Mead', options = {'maxiter': 5000})
    return parm_estimates


###Model 2 function###
def Model2(df07, df08, df_c07, df_c08):
    def payoff(parms):
        '''
        create a function that calculate the parameter of model 2
        the inputs are the variable names, all must be in df format

        Model 2:
        f(b; t) = delta*num_stations_buyer_bm * population_target_tm 
                   + alpha * corp_owner_buyer_bm * population_target_tm 
                   + gamma * HHI_tm
                   + beta * distancebtm + error_btm
        f (b,t) + pbt -  ≥ f(b, tdot) - pb,tdot
        '''
        delta, alpha, gamma, beta = parms
     
        f07 = delta * df07['num_stations_buyer']*df07['population_target'] + alpha * df07['corp_owner_buyer'] * df07['population_target'] + gamma *df07['hhi_target'] + beta *df07['distance'] + df07['price_mil']
        f08 = delta * df08['num_stations_buyer']*df08['population_target'] + alpha * df08['corp_owner_buyer'] * df08['population_target'] + gamma *df08['hhi_target'] + beta *df08['distance'] + df08['price_mil']
        
        f_count07 = delta * df_c07['num_stations_buyer']*df_c07['population_target'] + alpha * df_c07['corp_owner_buyer'] * df_c07['population_target'] + gamma *df_c07['hhi_target'] + beta *df_c07['distance'] + df_c07['price_mil']
        f_count08 = delta * df_c08['num_stations_buyer']*df_c08['population_target'] + alpha * df_c08['corp_owner_buyer'] * df_c08['population_target'] + gamma *df_c08['hhi_target'] + beta *df_c08['distance'] + df_c08['price_mil']
      
        a07=[]
        for i in len(df07[df07.buyer_id <= 989]):
            for j in len(df07[df07.buyer_id <= 989]):
                if i==j & f07 > f_count07:
                    a= 1
                    a07.append(a)      
        a08=[]
        for i in len(df08[df08.buyer_id >= 989]):
            for j in len(df08[df08.buyer_id >= 989]):
                if i==j & f08 > f_count08:
                    a= 1
                    a08.append(a)        
        
        joinedlist = a07 + a08
        #
        neg_total_score = -joinedlist.sum()
        return neg_total_score
    
    #Initial Guess
    guess = (0.5, 0.5, 0.5, 0.5)
    #calculate the parameter estimates
    #parm_estimates = differential_evolution(payoff, set_bound)
    parm_estimates = minimize(payoff, guess, method = 'Nelder-Mead', options = {'maxiter': 5000})
    return parm_estimates
##################################################################


##################################################################
###get the counterfactual dataset###
# and #
###calculate the distance and create sub dataset###

#column names for the buyer variables and target variables
buyer_colnames = ['year', 'buyer_id', 'buyer_lat', 'buyer_long', 'num_stations_buyer','corp_owner_buyer']
target_colnames = ['target_id', 'target_lat', 'target_long', 'price', 'price_mil', 'hhi_target', 'population_target', 'population_target_mil']

#get the counterfactual df by year
df_c2007=counterfactual_df_of_year(df, 2007, buyer_colnames, target_colnames)
df_c2008=counterfactual_df_of_year(df, 2008, buyer_colnames, target_colnames)

#Calculate distances
df_c2007=distances_of_year(df_c2007,2007)
df_c2008=distances_of_year(df_c2008,2008)

df_c=pd.concat([df_c2007, df_c2008], axis=0).reset_index()

df2007=distances_of_year(df,2007)
df2008=distances_of_year(df,2008)
df=pd.concat([df2007, df2008], axis=0).reset_index()
##################################################################


##################################################################
#Estimate the model and get the result

#Run the models
Model1_fit=Model1(df2007, df2008, df_c2007, df_c2008)

Model2_fit=Model2(df2007, df2008, df_c2007, df_c2008)


#print the parameter estimate for model 1
model1_list = ['alpha', 'beta']
model2_list = ['delta', 'alpha', 'gamma', 'beta']


print('Model 1 Coefficient Estimates')
for x, y in zip(model1_list, Model1_fit.x.tolist()):
    print(x, y, sep='\t\t')
print(" ")
print('Model 2 Coefficient Estimates')
for x, y in zip(model2_list, Model2_fit.x.tolist()):
    print(x, y, sep='\t\t')
