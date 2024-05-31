import pyclustering
from pyclustering.cluster.kmeans import kmeans as pyKMeans
from pyclustering.cluster.kmedoids import kmedoids as pyKMedoids
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.utils import calculate_distance_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
import numpy as np


np.warnings = warnings



def KMeans(X, custom_distance, n_clusters=2) :
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

def KMedoids(distance_matrix, custom_distance, n_clusters = 2) :
    rng = np.random.default_rng()
    l = np.arange(0, len(distance_matrix))
    np.random.shuffle(l)
    initial_medoids = l[:n_clusters]
    print(initial_medoids)
    k_medoids_instance = pyclustering.cluster.kmedoids.kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
    k_medoids_instance.process()
    clusters = k_medoids_instance.get_clusters()
    encoding = k_medoids_instance.get_cluster_encoding()
    encoder = cluster_encoder(encoding, clusters, distance_matrix)
    labels = encoder.set_encoding(0).get_clusters()
    print(labels)
    return (labels, np.nan)


def elbow_criteria(X, clustering_func, custom_distance) :
    sse = {}
    for k in range(2, 12):
        #kmeans = KMeans(n_clusters=k, max_iter=1000, n_init='auto').fit(X, sample_weight=w)
        kmeans = clustering_func(X, custom_distance, n_clusters=k)
        #X["clusters"] = kmeans.labels_
        #print(X["clusters"])
        sse[k] = kmeans[1]
    plt.xticks([i for i in range(2, 15)])
    plt.xlabel('k = количество кластеров')
    plt.legend('W(k) = within-cluster distance')
    plt.plot(list(sse.keys()), list(sse.values()))




def silhouette_criteria(X, clustering_func, custom_distance) :
    silhouettes = []
    for i in range(2, 15) :
        #kmeans = KMeans(n_clusters=i, n_init='auto').fit(X)
        kmeans = clustering_func(X, custom_distance, n_clusters=i)
        label = kmeans[0]
        silhouettes.append(silhouette_score(X, label, metric=custom_distance))
    plt.xticks([i for i in range(2, 15)])
    plt.xlabel('k = количество кластеров')
    plt.legend('s(k) = silhouette_score')
    plt.plot([i for i in range(2, 15)], silhouettes)



