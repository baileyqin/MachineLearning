import json
import random
import numpy as np


def cluster_points(X, mu):
    """
    Distribute data points into clusters.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - mu: A list of K cluster centers, each elements is a list of 2

    Returns:
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.
    """
    """
    compute distances between centers and points
    smallest distance goes to that cluster
    """
    clusters = {}

    for i in range(len(mu)):
        clusters[i] = []

    for i in range(len(X)):
        dis = np.zeros(len(mu))
        Xval = np.array(X[i])
        for j in range(len(mu)):
            muval = np.array(mu[j])
            dis[j] = np.linalg.norm(Xval-muval)
        closest = np.argmin(dis)
        clusters[closest].append(X[i])
    # you need to fill in your solution here
    return clusters


def reevaluate_centers(mu, clusters):
    """
    Update cluster centers.

    Inputs:
    - mu: A list of K cluster centers, each elements is a list of 2
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.

    Returns:
    - newmu: A list of K updated cluster centers, each elements is a list of 2
    """
    newmu = mu

    for i in range(len(mu)):
        if len(clusters[i]) == 0:
            newmu[i] = mu[i]
        A = np.array(clusters[i])
        asum = np.zeros(2)
        for j in range(len(A)):
            asum += A[j]
        avg =  asum/len(A)
        newmu[i] = avg.tolist()

    # you need to fill in your solution here

    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    random.seed(100)
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)

    return(mu, clusters)


def kmeans_clustering():
    # load data
    with open('hw4_circle.json', 'r') as f:
        data_circle = json.load(f)
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    clusters_all = {}
    for K in [2, 3, 5]:
        key = 'blob_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_blob, K)
        key = 'circle_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_circle, K)

    return mu_all, clusters_all


def main():
    mu_all, clusters_all = kmeans_clustering()

    print('K-means Cluster Centers:')
    for key, value in mu_all.items():
        print('\n%s:'% key)
        print(np.array_str(np.array(value), precision=4))

    with open('kmeans.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'clusters': clusters_all}, f_json)


if __name__ == "__main__":
    main()