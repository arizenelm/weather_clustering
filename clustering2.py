# %%
import pandas as pd
import numpy as np
import scipy.stats as scs
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
import scipy
import pyclustering
from pyclustering.cluster.kmeans import kmeans as pyKMeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

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
df_weather['month'] = df_weather['date'].dt.month - 1
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
weather_ids = pd.read_csv('data/weather_ids.csv', index_col=0, header=None, squeeze=True);
weather_ids.head()

# %%
region_ids = sorted(df_weather['region_ids'].unique())

# %%
per_month_clusters = {r : {m : 0 for m in range(12)} for r in region_ids}

# %%
for m in range(12) :
    # Филтр по месяцу
    df_weather_m = df_weather[df_weather['month'] == m][['region_ids', 'conditions_ids']]

    # Частоты погодных явлений для каждого района по фиксированному месяцу
    regions_dict_m = {r : {s : 0 for s in weather_ids.index} for r in region_ids}
    for index, row in df_weather_m.iterrows() :
        s = row['conditions_ids']
        regions_dict_m[row['region_ids']][s] += 1

    X = pd.DataFrame.from_dict(regions_dict_m, orient='index')
    clusters = KMeans(n_clusters=2, init='k-means++', max_iter=10000, random_state=5, n_init='auto').fit(preprocessing.normalize(X, axis=0))
    #clusters = DBSCAN().fit(preprocessing.normalize(X, axis=0))


    i = 0
    for r in region_ids :
        per_month_clusters[r][m] = clusters.labels_[i]
        i += 1

# %%
df_weather_pm = pd.DataFrame.from_dict(per_month_clusters, orient='index')

# %%
d_mean = np.zeros(12)
for j in range(12) :
    d_temp = 0
    for i1, i2 in product(region_ids, region_ids) :
        d_temp += int(df_weather_pm.iloc[i1, j] == df_weather_pm.iloc[i2, j])
    d_mean[j] = (d_temp + 0.01) / len(region_ids)**2

# %%
M = len(d_mean)
w = np.array([(1 / M) / d for d in d_mean])
w /= np.linalg.norm(w)
w

# %%
weights = w

# %%
weightened_hamming_distance = lambda x, y : scipy.spatial.distance.hamming(x, y)

# %%
hamming_distance = lambda x, y : scipy.spatial.distance.hamming(x, y)

# %%
custom_distance = weightened_hamming_distance

# %%
X = df_weather_pm.copy()

# %%
import warnings
np.warnings = warnings

# %%
distance_matrix = np.zeros((len(region_ids), len(region_ids)))

# %%
for i1, i2 in product(range(len(region_ids)), range(len(region_ids))) :
    distance_matrix[i1, i2] = distance_matrix[i2, i1] = custom_distance(X.iloc[i1], X.iloc[i2])

# %%
import kmedoids

# %%
def KMedoids(dist_matrix, custom_distance, n_clusters=2) :
    kmedoids_instance = kmedoids.KMedoids(n_clusters=n_clusters, method='fasterpam')
    kmedoids_instance.fit(dist_matrix)
    return (kmedoids_instance.labels_, kmedoids_instance.inertia_)

# %%

def silhouette_criteria(X, clustering_func, distance_matrix) :
    silhouettes = []
    for i in range(2, 15) :
        #kmeans = KMeans(n_clusters=i, n_init='auto').fit(X)
        kmeans = clustering_func(distance_matrix, custom_distance, n_clusters=i)
        label = kmeans[0]
        silhouettes.append(silhouette_score(X, label, metric=custom_distance))
    plt.xticks([i for i in range(2, 15)])
    plt.plot([i for i in range(2, 15)], silhouettes)

# %%
silhouette_criteria(X, KMedoids, distance_matrix)

# %%
clustering_utils.elbow_criteria(distance_matrix, KMedoids, np.nan)

# %%
kmeans = clustering_utils.KMedoids(distance_matrix, custom_distance, 4)

# %%
df_regions_clustered = df_weather_pm.copy()
df_regions_clustered['cluster_id'] = kmeans[0]
df_regions_clustered['region_name'] = df_region_ids['region']
# for i in range(len(df_regions1)) :
#     df_regions1['region_name'][i] = df_region_ids[i]
df_regions_clustered.head()

# %%
df_regions_clustered.to_csv('region2.csv')

# %%
silhouette_score(X, kmeans[0], metric=custom_distance)

# %%
dbscan = DBSCAN(metric=custom_distance).fit(X)

# %%
silhouette_score(X, dbscan.labels_, metric=custom_distance)

# %%
(dbscan.labels_)

# %%
#df_regions_clustered['cluster_id'] = dbscan.labels_
#df_regions_clustered.to_csv('region2.csv')


