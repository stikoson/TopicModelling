from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import rand_score
import numpy as np

def avg_dist_within(X, clustering):
    clusters = set(clustering)
    avg_dists = []
    n_elements = []
    
    for c in clusters:
        Y = [X[i] for i in range(len(clustering)) if clustering[i] == c]
        pwdY = pairwise_distances(Y)
        dists = pwdY[np.triu_indices(len(Y), k = 1)]
        avg_dists.append(np.mean(dists))
        n_elements.append(len(Y))
        
    avg_dist = np.average(avg_dists, weights = n_elements) / max(avg_dists)
    return avg_dist

def eval_clustering(X, labels, true_labels = []):

    sil_score = silhouette_score(X, labels, metric = 'euclidean')
    average_distance_within = avg_dist_within(X, labels)
    rand_sc = rand_score(true_labels, labels) if len(true_labels) != 0 else -1
    
    return([sil_score, average_distance_within, rand_sc])
