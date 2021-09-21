#!/usr/bin/env python
# coding: utf-8

# # PS 3. Iddrisu Kambala Mohammed

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_excel('jamesfearon.xls') ##conflict data from Fearon 
df


# In[2]:


df.sum


# In[ ]:





# In[3]:


x1 = df [['durest', 'year']].groupby('year').mean()
x2 = df [['dth', 'year']].groupby('year').mean()/10000 ##diving mean deaths by 10k to "normalise" it


# In[4]:


f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)
plt.plot(x1, linestyle = 'solid', color='blue', linewidth=2, markersize=1, label='duration')
font_1 = {'family':'serif','color':'black','size':18}
font_2 = {'family':'serif','color':'darkred','size':15}
plt.xlabel("years, 1945 - 2000", fontdict = font_2)
plt.ylabel("Average Duration of Wars", fontdict = font_2)
plt.title("Average Duration of Wars in Progress", fontdict = font_1, loc = 'center')
plt.legend()
plt.savefig('fig1.png')
plt.show()


# In[5]:


f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)
plt.plot(x2, linestyle = 'solid', color='red', linewidth=2, markersize=1, label='deaths')
font_1 = {'family':'serif','color':'black','size':18}
font_2 = {'family':'serif','color':'darkred','size':15}
plt.xlabel("years, 1945 - 2000", fontdict = font_2)
plt.ylabel("deaths", fontdict = font_2)
plt.title("Average Number of Deaths", fontdict = font_1, loc = 'center')
plt.legend()
plt.savefig('fig2.png')
plt.show()


# In[6]:


f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)
plt.plot(x1, linestyle = 'solid', color='blue', linewidth=2, markersize=1, label='duration')
plt.plot(x2, linestyle = 'solid', color='red', linewidth=2, markersize=1, label='deaths')
font_1 = {'family':'serif','color':'black','size':18}
font_2 = {'family':'serif','color':'darkred','size':15}
plt.xlabel("years, 1945 - 2000", fontdict = font_2)
plt.ylabel("Avg Duration of Wars & Deaths", fontdict = font_2)
plt.title("Average Duration of Wars in Progress & Number of Deaths", fontdict = font_1, loc = 'center')
plt.legend()
plt.savefig('fig3.png')
plt.show()


# In[7]:


jedwab = pd.read_excel('agg_patterns_gh.xlsx') #transportation and urbanisation data from Jedwab & Moradi
jedwab
jedwab.sum 


# In[8]:


x3 = jedwab.numcities_ghana
x4 = jedwab.urbrate_ghana
y3 = jedwab.year
s1mask = np.isfinite(x3)
s2mask = np.isfinite(x4)


# In[ ]:





# In[9]:


f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)
font_1 = {'family':'serif','color':'black','size':18}
font_2 = {'family':'serif','color':'darkred','size':15}
fig,ax = plt.subplots()
ax.plot(y3[s1mask], x3[s1mask], color="gray", marker="o", label='Number of Cities')
ax.set_xlabel("year",fontdict = font_2)
ax.set_ylabel("Number of Cities",color="red",fontdict = font_2)
ax2=ax.twinx()
ax2.plot(y3[s2mask], x4[s2mask],color="orange",marker="o", label='Urabisation Rate')
ax2.set_ylabel("Urbanisation Rate",color="blue",fontsize=14)
plt.title("Urbanisation (Ghana)", fontdict = font_1, loc = 'center')

plt.legend()
plt.savefig('fig4.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




