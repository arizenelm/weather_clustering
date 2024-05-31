# %%
import pandas as pd
import numpy as np
import scipy.stats as scs
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product

import scipy.spatial

import pyclustering
from pyclustering.cluster.kmeans import kmeans as pyKMeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


import clustering_utils

# %%
df_weather = pd.read_csv('data/weather.csv', index_col=0)
df_region_ids = pd.read_csv('data/regions.csv')

# %%
df_weather.head()

# %%
from ast import literal_eval

# %%
df_weather = df_weather[['date', 'region_ids', 'conditions_ids']]
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['region_ids'] = df_weather['region_ids'].apply(literal_eval)
df_weather['conditions_ids'] = df_weather['conditions_ids'].apply(literal_eval)
df_weather.head()

# %%
df_weather = df_weather[df_weather['conditions_ids'].apply(lambda x : len(x) > 0)]

# %%
df_weather.head()

# %%
df_weather['month'] = df_weather['date'].dt.month
df_weather.drop('date', axis=1, inplace=True)
df_weather.head()

# %%
df_weather = df_weather.explode('region_ids', ignore_index=True)
df_weather = df_weather.explode('conditions_ids', ignore_index=True)

# %%
print(len(df_weather))
df_weather.dropna(inplace=True)
print(len(df_weather))
df_weather.head()

# %%
weather_ids = pd.read_csv('data/weather_ids.csv', index_col=0, header=None).squeeze()
weather_ids

# %%
cond_count = len(weather_ids)
month_cond_labels = [str(i) + '_' + str(j) for (i, j) in product(range(1, 13), range(1, cond_count + 1))]

# %%
df_regions1 = pd.DataFrame(columns=['region_id'] + month_cond_labels)

# %%
print(cond_count * 12)
df_regions1.head()

# %%
regions_dict1 = {region : {s : 0 for s in month_cond_labels} for region in df_region_ids.index}

# %%
sorted(df_weather['conditions_ids'].unique())

# %%
for index, row in df_weather.iterrows() :
    s = str(row['month']) + '_' + str(row['conditions_ids'])
    regions_dict1[row['region_ids']][s] += 1

# %%
df_regions1 = pd.DataFrame.from_dict(regions_dict1, orient='index')

# %%
df_regions1.head()

# %%
df_regions1 = df_regions1.astype(float)

# %%
df_regions1.iloc[33] *= 0.3

# %%
X = df_regions1.copy()
X.head()

# %%
N = len(df_regions1)
d_mean = np.zeros(len(weather_ids))
d_mean_m = np.zeros(len(weather_ids))
for j in range(len(weather_ids)) :
    d_temp = 0
    d_temp_m = 0
    for i1, i2 in product(range(N), range(N)) :
        for m in range(12) :
            d_temp += (df_regions1.iloc[i1, m * len(weather_ids) + j] - df_regions1.iloc[i2, m * len(weather_ids) + j])**2
            d_temp_m += (df_regions1.iloc[i1, m * len(weather_ids) + j] - df_regions1.iloc[i2, m * len(weather_ids) + j])
    d_mean[j] = ((d_temp / N**2) + 0.01)
    d_mean_m[j] = ((d_temp_m / N**2) + 0.01)

# %%
d_mean_full = np.zeros(len(weather_ids) * 12)
d_mean_full_m = np.zeros(len(weather_ids) * 12)
for j in range(len(weather_ids) * 12) :
    d_temp = 0
    d_temp_m = 0
    for i1, i2, in product(range(N), range(N)) :
        d_temp += (df_regions1.iloc[i1, j] - df_regions1.iloc[i2, j])**2
        d_temp_m += (df_regions1.iloc[i1, j] - df_regions1.iloc[i2, j])
    d_mean_full[j] = (d_temp / N**2) + 0.01
    d_mean_full_m[j] = (d_temp_m / N**2) + 0.01

# %%
w = np.array([(1.0) / (d_mean[i % (12)]) for i in range(12 * len(d_mean))])
w_m = np.array([(1.0) / (d_mean_m[i % 12]) for i in range(12 * len(d_mean))])
w /= np.sum(w)
w_m /= np.sum(w_m)

# %%
w_full = np.array([(1.0) / d for d in d_mean_full])
w_full_m = np.array([(1.0) / d for d in d_mean_full_m])
w_full /= np.sum(w_full)
w_full_m /= np.sum(w_full_m)

# %%
temp = [100, 100, 100, 100, 100, 1, 100, 2, 100, 100, 1, 1, 2, 100, 1, 1, 1, 1]
w_custom = np.array([(1.0) / (temp[i % len(temp)]) for i in range(12 * len(d_mean))])
w_custom /= np.sum(w_custom)

# %%
weather_ids

# %%
weights = w_custom

# %%
euclidian_weightned_distance = lambda x, y : scipy.spatial.distance.euclidean(x, y, weights)

# %%
manhattan_weightned_distance = lambda x, y : scipy.spatial.distance.cityblock(x, y, weights)

# %%
cosine_weightned_distance = lambda x, y : scipy.spatial.distance.cosine(x, y, weights)

# %%
#custom_distance = lambda x, y : scipy.spatial.distance.euclidean(x, y)
custom_distance = cosine_weightned_distance

# %%
clustering_utils.silhouette_criteria((X), clustering_func=clustering_utils.KMeans, custom_distance=custom_distance)
plt.legend(('s(k) - silhouette coef'))
plt.ylabel('s = коэффициент силуэта')
plt.show()

# %%
clustering_utils.elbow_criteria(X, clustering_func=clustering_utils.KMeans, custom_distance=custom_distance)
plt.ylabel('W = внутрикластерное расстояние')
plt.legend('W')

# %%
#kmeans = KMeans(n_clusters=2).fit_predict((X))
kmeans = clustering_utils.KMeans((X), custom_distance, n_clusters = 4)

# %%
df_regions_clustered = df_regions1.copy()
df_regions_clustered['cluster_id'] = kmeans[0]
df_regions_clustered['region_name'] = df_region_ids['region']
# for i in range(len(df_regions1)) :
#     df_regions1['region_name'][i] = df_region_ids[i]
df_regions_clustered.head()

# %%
df_regions_clustered.to_csv('regions1.csv')

# %%
labels = df_regions_clustered['cluster_id']

# %%
sil_score = silhouette_score((X), labels, metric=custom_distance)
sil_score

# %%
dbscan = DBSCAN(metric=euclidian_weightned_distance).fit(X)

# %%
df_regions_clustered_temp = df_regions_clustered[df_regions_clustered['cluster_id'] == 0].drop(['cluster_id', 'region_name'], axis=1)

# %%
df_weather_clusters = df_weather.copy()
df_weather_clusters['cluster_id'] = [np.nan] * len(df_weather_clusters)
df_weather_clusters['condition_name'] = [weather_ids[i] for i in df_weather_clusters['conditions_ids']]
df_weather_clusters['condition_name'] = pd.Categorical(df_weather_clusters['condition_name'], list(df_weather_clusters['condition_name'].unique()))

# %%
for i in range(len(df_weather_clusters)) :
    r = df_regions_clustered.iloc[df_weather_clusters.iloc[i]['region_ids']]['cluster_id']
    df_weather_clusters.at[i, 'cluster_id'] = r
    #print(df_regions_clustered.iloc[i]['cluster_id'])

# %%
df_weather_clusters.head()

# %%
stop_list = ['гололед, изморозь', "снегопад, снежный покров", "заморозки", "ветер", "дожди, ливни", "гроза", "паводки, подтопления, наводнения", "град", "лавины"]

# %%
interesting = ['мороз', 'лавины', 'сель', 'вулканическая активность', 'пылевая буря', 'ураган', 'смерч']

# %%
df_weather_clusters1 = df_weather_clusters[df_weather_clusters['condition_name'].isin(interesting)].dropna()

# %%
df_weather_clusters1['condition_name'] = pd.Categorical(df_weather_clusters1['condition_name'], 
                                                        categories=['лавины', 'сель', 
                                                                            "вулканическая активность", 
                                                                            "мороз", "пылевая буря",
                                                                            "ураган", "смерч"])

# %%
def get_smth(i) :
    return df_weather_clusters[df_weather_clusters['cluster_id'] == i]['conditions_ids']

# %%
sns.histplot(get_smth(float(0)), stat='probability', discrete=True)
plt.xticks(rotation=90)
plt.ylim(0, 0.2)

# %%
sns.histplot(get_smth(float(1)), discrete=True, stat='probability')
plt.xticks(rotation=90)
plt.ylim(0, 0.2)

# %%
sns.histplot(get_smth(float(2)), discrete=True, stat='probability')
plt.xticks(rotation=90)
plt.ylim(0, 0.2)

# %%
sns.histplot(get_smth(float(3)), discrete=True, stat='probability')
plt.xticks(rotation=90)
plt.ylim(0, 0.2)


