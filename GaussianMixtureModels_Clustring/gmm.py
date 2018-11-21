import json
import random
import numpy as np

def norm(X,mu,cov):
    """
    calculates the normal distribution probability of a point being at X
    :param X: position of point
    :param mu:  average of gaussian
    :param cov: covariance
    :return:
    prob: probability
    """
    X = np.array(X)
    mu = np.array(mu)
    cov = np.array(cov)
    cov = cov.reshape((2, 2))
    cons = 1.0/np.linalg.det(2*np.pi*cov)
    prob = cons * np.exp((-1.0/2.0)*np.matmul(np.matmul(np.transpose(X-mu),np.linalg.inv(cov)), (X-mu)))
    return prob

def compute_gamma(X, pi, mu, cov):
    """
    computes gammank a 2d array
    X all positions of n points
    pi: mixing coefficients of gaussians
    mu: averages for all gausians
    cov: covarience for all gaussians

    returns gam:gammank a 2d array
    """
    K = len(mu)
    N = len(X)
    gaus = np.zeros(K)
    gam = np.zeros((N, K))
    for n in range(N):
        gaussum = 0.0
        for k in range(K):
            gaus[k] = pi[k] * norm(X[n], mu[k], cov[k])
            gaussum += gaus[k]
        for k in range(K):
            gam[n][k] = gaus[k]/gaussum
    gam = gam.tolist()
    return gam

def update_parm(gam, X, K):
    """
    :param gam: gamma values
    :return:
    pi: the weights
    mu: the average positions
    cov: the covarience
    """
    X = np.array(X)
    N, M = X.shape
    pi = np.zeros(K)
    gamsum = 0.0
    for k in range(K):
        for n in range(N):
            gamsum += gam[n][k]
            pi[k] += gam[n][k]

    for k in range(K):
        pi[k] = pi[k]/gamsum

    pi = pi.tolist()

    mu = np.zeros((K, M))
    for k in range(K):
        musumtop = np.zeros(M)
        musumbot = 0.0
        for n in range(N):
            musumtop += gam[n][k] * X[n]
            musumbot += gam[n][k]
        mu[k] = (1.0/musumbot) * musumtop

    cov = []
    for k in range(K):
        covsumtop = np.zeros((M, M))
        covsumbot = 0.0
        for n in range(N):
            covsumtop += gam[n][k] * np.outer((X[n] - mu[k]), (X[n] - mu[k]))
            covsumbot += gam[n][k]
        temp_cov = (1.0 / covsumbot) * covsumtop
        cov.append(list(temp_cov.reshape(4)))
    mu = mu.tolist()
    return pi, mu, cov

def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))

    ### you need to fill in your solution starting here ###
    """
    0 guess parameters
    1 compute probablities gammank with parameters
    2 update parameters with new gammank
    3 go back to 1
    """

    # Run 100 iterations of EM updates
    for t in range(100):
        gam = compute_gamma(X, pi, mu, cov)
        pi, mu, cov = update_parm(gam, X, K)

    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()