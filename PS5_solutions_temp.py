#import numpy as np
import pandas as pd
#from scipy.optimize import minimize
from geopy.distance import geodesic
from scipy.optimize import differential_evolution

df = pd.read_csv(r"C:\Users\haimiti.aerfate\Desktop\radio_merger_data.csv")

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
    for i in range(len(x["buyer_id"])):
        for j in range(len(x["buyer_id"])):
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
def Model1(num_stations_buyer, population_target, corp_owner_buyer, distance,
           num_stations_buyer_c, population_target_c, corp_owner_buyer_c, distance_c):

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
        
        f = num_stations_buyer*population_target + alpha * corp_owner_buyer * population_target + beta *distance
        
        f_count = num_stations_buyer_c*population_target_c + alpha * corp_owner_buyer_c * population_target_c + beta *distance_c
        
        score = (f >= f_count)
        
        #
        neg_total_score = -score.sum()
        return neg_total_score
    
    #Initial Guess
    set_bound = [(-1,1),(-1,1)]
    #calculate the parameter estimates
    parm_estimates = differential_evolution(payoff, set_bound)
    return parm_estimates


###Model 2 function###
def Model2(num_stations_buyer, population_target, corp_owner_buyer, distance, hhi_target, price_mil,
           num_stations_buyer_c, population_target_c, corp_owner_buyer_c, distance_c, hhi_target_c, price_mil_c):

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

        f = delta * num_stations_buyer * population_target + alpha * corp_owner_buyer * population_target + gamma * hhi_target + beta *distance + price_mil
        
        f_count = delta * num_stations_buyer_c * population_target_c + alpha * corp_owner_buyer_c * population_target_c + gamma * hhi_target_c + beta *distance_c + price_mil_c
        
        #
        score = (f >= f_count)
        
        #
        neg_total_score = -score.sum()
        return neg_total_score
    
    #Initial Guess
    set_bound = [(-1,1),(-1,1),(-1,1),(-1,1)]
    #calculate the parameter estimates
    parm_estimates = differential_evolution(payoff, set_bound)
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
c2007=counterfactual_df_of_year(df, 2007, buyer_colnames, target_colnames)
c2008=counterfactual_df_of_year(df, 2008, buyer_colnames, target_colnames)

#Calculate distances
c2007=distances_of_year(c2007,2007)
c2008=distances_of_year(c2008,2008)

df_c=pd.concat([c2007, c2008], axis=0).reset_index()

df2007=distances_of_year(df,2007)
df2008=distances_of_year(df,2008)
df=pd.concat([c2007, c2008], axis=0).reset_index()
##################################################################


##################################################################
#Estimate the model and get the result

num_stations_buyer = df['num_stations_buyer']
population_target = df['population_target_mil']
hhi_target = df['hhi_target']
corp_owner_buyer = df['corp_owner_buyer']
distance = df['distance']
price_mil = df['price_mil']

num_stations_buyer_c = df_c['num_stations_buyer']
population_target_c = df_c['population_target_mil']
hhi_target_c = df_c['hhi_target']
corp_owner_buyer_c = df_c['corp_owner_buyer']
distance_c = df_c['distance']
price_mil_c = df_c['price_mil']

#Run the models
Model1_fit=Model1(num_stations_buyer, population_target, corp_owner_buyer, distance,
           num_stations_buyer_c, population_target_c, corp_owner_buyer_c, distance_c)

Model2_fit=Model2(num_stations_buyer, population_target, corp_owner_buyer, distance, hhi_target, price_mil,
           num_stations_buyer_c, population_target_c, corp_owner_buyer_c, distance_c, hhi_target_c, price_mil_c)


#print the parameter estimate for model 1
model1_list = ['alpha', 'beta']
model2_list = ['delta', 'alpha', 'gamma', 'beta']


print('Model 1 Coefficient Estimates')
for x, y in zip(model1_list, Model1_fit.x.tolist()):
    print(x, y, sep='\t\t')
print(" ")
#print the parameter estimate for model 2
print('Model 2 Coefficient Estimates')
for x, y in zip(model2_list, Model2_fit.x.tolist()):
    print(x, y, sep='\t\t')




'''
I should have put the merged actual and conterfactual dataset
into one like this:
dfb=pd.merge(df2008, c2008, left_on='buyer_id', right_on='buyer_id', how='left')
   
Unforunately, I don't have time

I know my solution is incorrect, ideally i think it should look like this.
However, it doesn't work. 

def Model1(num_stations_buyer, population_target, corp_owner_buyer, distance,
           num_stations_buyer_c, population_target_c, corp_owner_buyer_c, distance_c):

    def payoff(parms):
        ''
        create a function that calculate the parameter of model 1
        the inputs are the variable names, all must be in df format

        Model 1:
        f(b; t) = num_stations_buyer_bm * population_target_tm 
                   + alpha * corp_owner_buyer_bm * population_target_tm 
                   + beta * distancebtm + error_btm
        f (b,t) + f (bdot, tdot)  ≥ f (bdot, t) + f(b, tdot)
        ''
        alpha, beta = parms
        
        f = num_stations_buyer*population_target + alpha * corp_owner_buyer * population_target + beta *distance
        
        f_count = num_stations_buyer_c*population_target_c + alpha * corp_owner_buyer_c * population_target_c + beta *distance_c
        
        score=[]
        for index1, i in enumerate(corp_owner_buyer, start=0):
            for index2, j in enumerate(corp_owner_buyer_c, start=0):
                if i==j & f > f_count:
                    aa= 1
                    score.append(aa)
        
        neg_total_score = -(score.sum())
        return neg_total_score

    #Initial Guess
    set_bound = [(-1,1),(-1,1)]
    #calculate the parameter estimates
    parm_estimates = differential_evolution(payoff, set_bound)
    return parm_estimates
'''