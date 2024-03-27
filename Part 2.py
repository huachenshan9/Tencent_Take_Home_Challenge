#!/usr/bin/env python
# coding: utf-8

# In[46]:


get_ipython().system('pip install plotly')


# In[47]:


import numpy as np
import pandas as pd

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
#import scikitplot as skp


# In[48]:


from sklearn.datasets import make_classification  
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import StackingClassifier  
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import *
from sklearn.metrics import make_scorer





import warnings
#warnings.filterwarnings('ignore')


# In[49]:


df = pd.read_csv(r'C:\Users\Shan Huachen\Desktop\topline_metrics.csv', encoding='ANSI')#please change to the own file address on your computer before running it.
df.head()


# In[50]:


df.eval('gap = D1 - D7', inplace = True) #Calculate the time from just registering an account to being grouped for players.
df


# In[51]:


#Topline Metrics
#Overall
df1 = df[df['Platform'] == 'ALL']
df1


# In[52]:


df1['Date'] = pd.to_datetime(df1['Date'])


# In[53]:


df1 = df1.sort_values(by = 'Date')


# In[43]:


x = df1['Date']
y = df1['gap']
 
plt.plot(x, y, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("gap_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[8]:


x = df1['Date']
y = df1['TRU']
 
plt.plot(x, y, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("TRU_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[9]:


df["Platform"].value_counts()


# In[ ]:


#All：315288，IOS: 141790, Android: 173498 


# In[54]:


df1_Android = df[df['Platform'] == 'Android(All)']


# In[55]:


df1_Android


# In[12]:


df1_Android['Date'] = pd.to_datetime(df1_Android['Date'])


# In[13]:


df1_Android = df1_Android.sort_values(by = 'Date')


# In[14]:


x_Android = df1_Android['Date']
y_Android = df1_Android['DAU']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("DAY_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[29]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["TRU"],
        title = "Total Register User in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[30]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["DAU"],
        title = "Daily Active User in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[35]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["NRU"],
        title = "New Register User in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[32]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["Revenue"],
        title = "Revenue in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[57]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["Payer"],
        title = "Payer in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[36]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["UC Total Outflow (K)"],
        title = "UC Total Outflow (K) in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[37]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["D1"],
        title = "D1 in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[38]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["D7"],
        title = "D7 in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[28]:


fig = px.line(df1_Android,
        x = "Date",
        y = ["AOT(min)"],
        title = "AOT(min) in these time period",
        template = "plotly_dark"      
       )

fig.show()


# In[31]:


x_Android = df1_Android['Date']
y_Android = df1_Android['NRU']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("NRU_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[27]:


x_Android = df1_Android['Date']
y_Android = df1_Android['TRU']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("TRU_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[33]:


x_Android = df1_Android['Date']
y_Android = df1_Android['Revenue']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("Revenue_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[16]:


x_Android = df1_Android['Date']
y_Android = df1_Android['UC Total Outflow (K)']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("UC Total Outflow (K)_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[17]:


x_Android = df1_Android['Date']
y_Android = df1_Android['D1']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("D1_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[18]:


x_Android = df1_Android['Date']
y_Android = df1_Android['D7']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("D7_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[19]:


x_Android = df1_Android['Date']
y_Android = df1_Android['AOT(min)']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("AOT(min)_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[ ]:


x_Android = df1_Android['Date']
y_Android = df1_Android['Payer']
 
plt.plot(x_Android, y_Android, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("Payer_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[18]:


df1_IOS = df[df['Platform'] == 'IOS(All)']


# In[19]:


df1_IOS


# In[20]:


df1_IOS['Date'] = pd.to_datetime(df1_IOS['Date'])


# In[21]:


df1_IOS = df1_IOS.sort_values(by = 'Date')


# In[22]:


x_IOS = df1_IOS['Date']
y_IOS = df1_IOS['TRU']
 
plt.plot(x_IOS, y_IOS, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("TRU_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[29]:


x_IOS = df1_IOS['Date']
y_IOS = df1_IOS['DAU']
 
plt.plot(x_IOS, y_IOS, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("DAU_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[32]:


x_IOS = df1_IOS['Date']
y_IOS = df1_IOS['NRU']
 
plt.plot(x_IOS, y_IOS, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("NRU_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[34]:


x_IOS = df1_IOS['Date']
y_IOS = df1_IOS['Revenue']
 
plt.plot(x_IOS, y_IOS, color = 'blue', linewidth = 1)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("Revenue_Trend")  # add title
plt.show()

#change tolinechart
#check the output from Android and IOS


# In[ ]:


#insight analysis
#seasonality, Q2 is the peak season every year, with July being the lowest point. From July to the next July, there is a gradual increase, and the period between each year is relatively neutral.


# In[48]:


df1.dtypes


# In[5]:


df.describe().style.background_gradient(cmap="ocean_r")  # Beautify table output.


# In[6]:


null_df = pd.DataFrame({"Null Values": df.isnull().sum(),
                         "Percentage Null Values": (df.isnull().sum()) / (df.shape[0]) * 100
                         })

null_df


# In[10]:


df_frequency = pd.concat([df['Platform'],
                        df['TRU'],
                        df['DAU'],
                        df['NRU'],
                        df['Revenue'],
                        df['Payer'],
                        df['UC Total Outflow (K)'],
                        df['D1'],
                        df['D7'],
                        df['AOT(min)']],
                       axis=1)

df_frequency.head()


# In[13]:


# Shape the canvas

fig, ax = plt.subplots(ncols=8, figsize=(20,6))

sns.scatterplot(data=df_frequency,
                x="TRU",
                y="Revenue",
                hue="Platform",
                ax=ax[0])

sns.scatterplot(data=df_frequency,
                x="DAU",
                y="Revenue",
                hue="Platform",
                ax=ax[1])

sns.scatterplot(data=df_frequency,
                x="NRU",
                y="Revenue",
                hue="Platform",
                ax=ax[2])

sns.scatterplot(data=df_frequency,
              x="Payer",
              y="Revenue",
              hue="Platform",
              ax=ax[3])

sns.scatterplot(data=df_frequency,
              x="UC Total Outflow (K)",
              y="Revenue",
              hue="Platform",
              ax=ax[4])

sns.scatterplot(data=df_frequency,
              x="D1",
              y="Revenue",
              hue="Platform",
              ax=ax[5])

sns.scatterplot(data=df_frequency,
              x="D7",
              y="Revenue",
              hue="Platform",
              ax=ax[6])

sns.scatterplot(data=df_frequency,
              x="AOT(min)",
              y="Revenue",
              hue="Platform",
              ax=ax[7])
plt.show()


# In[11]:


df['Platform'].value_counts()


# In[12]:


pie, ax = plt.subplots(figsize=[8,6])
labels = ['ALL', 'Android(All)','IOS(All)','android','iOS','Android','IOS']
colors = ['r','b','g','k','w','y','m']
plt.pie(x = df['Platform'].value_counts(), autopct='%.2f%%', 
        explode=[0.02]*7, labels=labels, pctdistance=0.5, textprops={'fontsize': 14}, colors = colors)
plt.title('Label distribution of original dataset',fontsize=14)
plt.legend(loc='upper right')
plt.show()


# In[ ]:




