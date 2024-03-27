#!/usr/bin/env python
# coding: utf-8

# In[208]:


get_ipython().system('pip install pingouin')


# In[209]:


import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.weightstats import ttost_paired
from scipy import stats


# In[210]:


df = pd.read_csv(r'C:\\Users\\Shan Huachen\\Desktop\\user_profile.csv', encoding='ANSI') #please change it to your own file address before run it
df.head()


# In[211]:


df1_Android = df[df['platid'] == 1]


# In[212]:


df1_ios = df[df['platid'] == 0]


# In[213]:


df1_ios


# In[214]:


df1_Android


# In[215]:


df.eval('duration = killingcount/survivaltime', inplace = True) #计算从刚刚注册账号到被分组的玩家时间
df


# In[216]:


df.info()


# In[217]:


df1_ios.info()


# In[218]:



df['dtstatdate'] = pd.to_datetime(df['dtstatdate'])
df['variant'] = df['variant'].replace(['A', 'B', 'C', 'Control'], ['a', 'b', 'c','cont'])
df = df.rename(columns = {'variant':'design'})

df.info()


# In[219]:


df1_Android['dtstatdate'] = pd.to_datetime(df1_Android['dtstatdate'])
df1_Android['variant'] = df1_Android['variant'].replace(['A', 'B', 'C', 'Control'], ['a', 'b', 'c','cont'])
df1_Android = df1_Android.rename(columns = {'variant':'design'})

df1_Android.info()


# In[165]:


df1_ios['dtstatdate'] = pd.to_datetime(df1_ios['dtstatdate'])
df1_ios['variant'] = df1_ios['variant'].replace(['A', 'B', 'C', 'Control'], ['a', 'b', 'c','cont'])
df1_ios = df1_ios.rename(columns = {'variant':'design'})

df1_ios.info()


# In[73]:


df['dtstatdate'].unique()


# In[74]:


df['design'].unique()


# In[75]:


len(df['vgameappid'].unique())


# In[76]:


len(df['vopenid'].unique())


# In[77]:


group = df.groupby(['dtstatdate','design']).mean('ionlinetime')
group


# In[167]:


group = df1_ios.groupby(['dtstatdate','design']).mean('ionlinetime')
group


# In[220]:


group = df1_Android.groupby(['dtstatdate','design']).mean('ionlinetime')
group


# In[ ]:





# In[251]:


design_a = group.query('design == "a"')['killingcount']
design_b = group.query('design == "b"')['killingcount']
design_c = group.query('design == "c"')['killingcount']
design_cont = group.query('design == "cont"')['killingcount']


# In[ ]:





# In[252]:


sci_a = stats.ttest_rel(design_a, design_cont)
sci_a


# In[253]:


sci_b = stats.ttest_rel(design_b, design_cont)
sci_b


# In[254]:


sci_c = stats.ttest_rel(design_c, design_cont)
sci_c


# In[255]:


stats.levene(design_a, design_cont)


# In[256]:


stats.levene(design_b, design_cont)


# In[257]:


stats.levene(design_c, design_cont)


# In[258]:


ping_a = pg.ttest(design_a, design_cont, paired = True, correction = False)
ping_a


# In[259]:


ping_b = pg.ttest(design_b, design_cont, paired = True, correction = False)
ping_b


# In[260]:


ping_c = pg.ttest(design_c, design_cont, paired = True, correction = False)
ping_c


# ### Users can use the same method to compare other column variables (like iamount / itimes) between control group and treatment group

# In[ ]:




