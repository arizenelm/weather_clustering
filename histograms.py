# %%
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ast import literal_eval

# %%
df_weather = pd.read_csv('data/weather.csv', index_col=0)

# %%
df_weather.head()

# %%
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['region_ids'] = df_weather['region_ids'].apply(literal_eval)
df_weather['conditions_ids'] = df_weather['conditions_ids'].apply(literal_eval)
df_weather['conditions'] = df_weather['conditions'].apply(literal_eval)

# %%
df_weather_conditions = df_weather.explode(column='conditions')
df_weather_conditions.dropna(inplace=True)
df_weather_conditions.head()

# %%
h = sns.histplot(df_weather_conditions['conditions'], discrete=True)
plt.xticks(rotation=90)

# %%
df_weather_conditions = df_weather_conditions.explode('region_ids')
df_weather_conditions.dropna(inplace=True)

# %%
g = sns.FacetGrid(df_weather_conditions[['region_ids', 'conditions']], col="conditions", col_wrap=6, height=4, ylim=(0, 20))
g.map(sns.histplot, 'region_ids', discrete=True)


plt.show()