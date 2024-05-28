# %%
import pandas as pd
import numpy as np
import scipy.stats as scs
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product

import scipy.spatial
from yellowbrick.cluster import KElbowVisualizer

import pyclustering
from pyclustering.cluster.kmeans import kmeans as pyKMeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
weather_ids = pd.read_csv('data/weather_ids.csv', index_col=0, header=None, squeeze=True);
weather_ids.head()

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
for j in range(len(weather_ids)) :
    d_temp = 0
    for i1, i2 in product(range(N), range(N)) :
        for m in range(12) :
            d_temp += (df_regions1.iloc[i1, m * len(weather_ids) + j] - df_regions1.iloc[i2, m * len(weather_ids) + j])**2
    d_mean[j] = ((d_temp / N**2) + 0.01)

# %%
d_mean_full = np.zeros(len(weather_ids) * 12)
for j in range(len(weather_ids) * 12) :
    d_temp = 0
    for i1, i2, in product(range(N), range(N)) :
        d_temp += (df_regions1.iloc[i1, j] - df_regions1.iloc[i2, j])**2
    d_mean_full[j] = (d_temp / N**2) + 0.01

# %%
w = np.array([(1.0 / N) / d_mean[i % 12] for i in range(12 * len(d_mean))])
w /= np.linalg.norm(w)

# %%
w_full = np.array([(1.0 / N) / d for d in d_mean_full])
w_full /= np.linalg.norm(w_full)

# %%
weights = w

# %%
euclidian_weightned_distance = lambda x, y : scipy.spatial.distance.euclidean(x, y, weights)

# %%
manhattan_weightned_distance = lambda x, y : scipy.spatial.distance.cityblock(x, y, weights)

# %%
custom_distance = euclidian_weightned_distance

# %%
import warnings
np.warnings = warnings

# %%
def KMeansCustom(X, custom_distance, n_clusters=2) :
    metric = distance_metric(type_metric.USER_DEFINED, func=custom_distance)
    initial_centers = kmeans_plusplus_initializer(X, n_clusters, random_state=5).initialize()
    kmeans = pyKMeans(X, initial_centers=initial_centers, metric=metric)
    kmeans.process()
    pyClusters = kmeans.get_clusters()
    pyEncoding = kmeans.get_cluster_encoding()
    pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)
    pyLabels = pyEncoder.set_encoding(0).get_clusters()
    wce = kmeans.get_total_wce()
    return (pyLabels, wce)

# %%
def elbow_criteria(X, custom_distance) :
    sse = {}
    for k in range(2, 12):
        #kmeans = KMeans(n_clusters=k, max_iter=1000, n_init='auto').fit(X, sample_weight=w)
        kmeans = KMeansCustom(X, custom_distance, n_clusters=k)
        #X["clusters"] = kmeans.labels_
        #print(X["clusters"])
        sse[k] = kmeans[1]
    plt.xticks([i for i in range(2, 15)])
    plt.plot(list(sse.keys()), list(sse.values()))

# %%
def silhouette_criteria(X, custom_distance) :
    silhouettes = []
    for i in range(2, 15) :
        #kmeans = KMeans(n_clusters=i, n_init='auto').fit(X)
        kmeans = KMeansCustom(X, custom_distance, n_clusters=i)
        label = kmeans[0]
        silhouettes.append(silhouette_score(X, label, metric=custom_distance))
    plt.xticks([i for i in range(2, 15)])
    plt.plot([i for i in range(2, 15)], silhouettes)

# %%
silhouette_criteria(X, custom_distance)

# %%
elbow_criteria(X, custom_distance)

# %%
#kmeans = KMeans(n_clusters=2).fit_predict((X))
kmeans = KMeansCustom(X, custom_distance, n_clusters = 4)

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


