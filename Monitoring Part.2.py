#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


df = pd.read_excel(r'C:\Users\raflg\Downloads\Databases\Equipe.xls')
df


# # Visualisation

# In[3]:


df.loc[(df['nomSeance'] == 'Seance individuelle') | (df['nomSeance'] == 'Seance collective'), 'nomSeance'] = 'Seance'
#df.columns


# In[4]:


marqueurs = ['marqueur1', 'marqueur2', 'marqueur3', 'marqueur4', 'marqueur5', 'marqueur6', 'marqueur7', 'marqueur8', 
             'marqueur9', 'marqueur10', 'marqueur11', 'marqueur12', 'marqueur13', 'marqueur14', 'marqueur15',
             'marqueur16', 'marqueur17', 'marqueur18']

Joueurs = df.sportif.unique().tolist()


# In[5]:


agg_func = {'duree' : 'sum'}

for m in marqueurs:
    agg_func[m] = 'max'


# In[6]:


df_plot = df.groupby(['dateSeance', 'sportif', 'nomSeance', 'numsemaine'], as_index=False).agg(agg_func)
#df_plot


# In[7]:


joueurs_dp = widgets.Dropdown(description='Joueur :',
                           value=Joueurs[0],
                           options=Joueurs
                          )

marqueurs_dp = widgets.Dropdown(description='Marqueur :', 
                                value=marqueurs[0],
                                options=marqueurs
                               )

trace1 = go.Scatter(x=df_plot.loc[df_plot['sportif']==Joueurs[0],'dateSeance'], 
                    y=df_plot.loc[df_plot['sportif']==Joueurs[0], marqueurs[0]],
                    mode='lines+markers', fill='tozeroy', marker_color='#576DF3',
                    marker=dict(opacity=0.8,size=3), line=dict(width=1),
                    name='Seances')

trace2 = go.Bar(x=df_plot.loc[(df_plot['sportif']==Joueurs[0])&(df_plot['nomSeance']=='Match'),'dateSeance'],
                y=df_plot.loc[(df_plot['sportif']==Joueurs[0])&(df_plot['nomSeance']=='Match'), marqueurs[0]],
                marker_color='#576DF3',
                name='Matchs', opacity=1)

g = go.FigureWidget(data=[trace1,trace2], layout=go.Layout(barmode='overlay'))

def response(change):
    x1 = df_plot.loc[df_plot['sportif']==joueurs_dp.value,'dateSeance']
    y1 = df_plot.loc[df_plot['sportif']==joueurs_dp.value, marqueurs_dp.value]
    x2 = df_plot.loc[(df_plot['sportif']==joueurs_dp.value)&(df_plot['nomSeance']=='Match'), 'dateSeance']
    y2 = df_plot.loc[(df_plot['sportif']==joueurs_dp.value)&(df_plot['nomSeance']=='Match'), marqueurs_dp.value]
    
    
    with g.batch_update():
        g.data[0].x = x1
        g.data[0].y = y1
        g.data[1].x = x2
        g.data[1].y = y2
        g.layout.barmode = 'overlay'

joueurs_dp.observe(response)
marqueurs_dp.observe(response)

g.update_layout(width=1000, height=700,
                  yaxis=dict(range=[0,10]),
                  template='plotly_white',
                  autosize=False)

g.update_xaxes(tickangle=45, dtick='L1')

g.update_layout(xaxis=dict(rangeselector=dict(buttons=
                                              list([dict(count=14, label='2w', step='day'),
                                                    dict(count=1, label='1m', step='month'),
                                                    dict(count=2, label='2m', step='month'),
                                                    dict(count=6, label='6m', step='month'),
                                                    dict(count=1, label='1y', step='year'),
                                                    dict(step='all')])
                                             ), rangeslider=dict( visible=True), type="date"))

container = widgets.HBox([joueurs_dp, marqueurs_dp])
widgets.VBox([container, g])


# # Standardisation

# In[8]:


from sklearn.preprocessing import StandardScaler


# ### Standardisation normale à la moyenne

# In[9]:


X = df.iloc[:, 5:]

scaler = StandardScaler()
X_z = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_z, columns=marqueurs)
#df_scaled


# ### Standardisation à la médiane
# #### On ne peut pas calculer de moyenne pour des données qualitatives. Bien qu'elles soient retranscies en données quantitatives, le calcul de la médiane est préférable

# In[10]:


df_scaled_median = pd.DataFrame(columns=marqueurs)

for m in marqueurs:
    median_c = df[m].median()
    num = 0
    for i in df[m]:
        num += (i - median_c)**2
    denom = len(df) - 1
    var = num /denom
    c_std = np.sqrt(var)
    df_scaled_median[m] = (df[m] - median_c) / c_std
    
#df_scaled_median


# # Corrélation/Covariance
# ### Avec des données standardisées, la corrélation et la covariance sont identiques
# #### Pre-processing avant le Machine learning (ML). Permet d'avoir une première visualisation d'ensemble et d'assurer des critères avant toute technique de ML : avoir des marqueurs avec une corrélation, ne pas avoir trop de marqueurs avec une corrélation trop forte (1)

# ## Corr/Cov standardisation normale

# In[11]:


plt.figure(figsize=(15, 15))
sns.heatmap(df_scaled.corr(), annot=True, cmap='Blues')
plt.show()


# ## Corr/Cov standardisation à la médiane
# #### Les tables sont identiques car la médiane et la moyenne des marqueurs sont très proches

# In[12]:


plt.figure(figsize=(15, 15))
sns.heatmap(df_scaled_median.corr(), annot=True, cmap='Blues')
plt.show()

