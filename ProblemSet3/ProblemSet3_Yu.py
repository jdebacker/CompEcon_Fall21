import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ex

# Read in data from csv file
df = pd.read_csv('BankChurners.csv',sep=',') 

data_CA = df['Credit_Limit']

# hist distribution of Credit line 
plt.hist(data_CA, bins = 20, edgecolor = 'black') 

plt.ylabel('Number of customer in this credit line interval')
plt.xlabel('Credit line')
plt.title('Distribution of credit line')

plt.savefig('Distribution_of_CL')

# show the distribution of gender
import plotly.io as pio
fig1 = ex.pie(df,names='Gender',title='Distribution of gender')
pio.write_image(file='gender.png',fig = fig1)

# show the distribution of credit line for male and female customers
data_CA_M = df[df['Gender']=='M']['Credit_Limit']
data_CA_F = df[df['Gender']=='F']['Credit_Limit']

kwargs = dict(histtype='stepfilled', alpha = 0.3, bins=20)
plt.hist(data_CA_M, **kwargs, color='g')
plt.hist(data_CA_F, **kwargs, color='r')
plt.legend(['male','female'])
plt.title('Distribution of credit line for male and female')
plt.savefig('Distribution_of_CreditLine_Gender')

