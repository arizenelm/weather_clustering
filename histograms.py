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
df_weather['region_names'] = df_weather['region_names'].apply(literal_eval)

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
df_weather_conditions.head()

# %%
g = sns.FacetGrid(df_weather_conditions[['region_ids', 'conditions']] \
                  .query('(conditions == "лавины") | (conditions == "снегопад, снежный покров")'), 
                  col="conditions", col_wrap=3, height=4, ylim=(0, 0.35))
g.set_ylabels("Частота")
g.map(sns.histplot, 'region_ids', discrete=True, stat='probability')
g.axes[0].set_xlabel('Район', visible=True)
g.axes[1].set_xlabel('Район')

# %%
sns.histplot(df_weather_conditions['region_ids'], discrete=True)

# %%
df_weather.explode('region_names').groupby('region_names').count()['date'].sort_values(ascending=False).head(20)

# %%
g = sns.FacetGrid(df_weather_conditions[(df_weather_conditions['region_ids'] == 33) | (df_weather_conditions['region_ids'] == 41)], col="region_ids", col_wrap=2, height=5, aspect=1.5)
g.map(sns.histplot, 'conditions', discrete=True, stat='probability')
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]


