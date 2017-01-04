import numpy as np
from getdata import importdata
import kmeans as km
import mixture as em
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


"""
HMM on a mixture of 4 gaussians
Author: Valentin Thomas
Date: 23/10/2015

The transition probabilities, the means and covariances are all estimated with
the EM algorithm. Means and covariances are initialized with a K-means.

Variables:
    p: initial probability
    A: transition matrix
    K: number of clusters
    mu: matrix of the means of the gaussians
    sigma: matrix of the covariances of the gaussians

In the notes:
    q: latent variable (cluster)
    u: observations
    alpha_t: forward joint probability p(q_t, u_1..t)
    beta_t: backward conditional probability p(u_T..t+1|q_t)
    gamma_t: total conditional probabity p(q_t|u_1..T)
    chsi_t: total conditional transition probabity p(q_t, q_t+1|u_1..T)
"""

## Numerically stable helping functions

# Computes log(v*p)
def log_dot(v, log_p):
    log_v = np.log(v)
    max_log = max(log_v+log_p)
    log_ratio = log_v + log_p - max_log
    return max_log + np.log(sum(np.exp(log_ratio)))

# Computes log(A*p)
def log_matrix_dot(A, log_p):
    res = np.zeros(log_p.shape)
    for i in range(len(res)):
        res[i] = log_dot(A[i,:], log_p)
    return res

def log_max(v, log_p):
    log_v = np.log(v)
    max_log = max(log_v+log_p)
    return max_log


def log_matrix_max(A, log_p):
    res = np.zeros(log_p.shape)
    for i in range(len(res)):
        res[i] = log_dot(A[i,:], log_p)
    return res

## Message-Passing part

# alpha-beta recursion
def forward_backward(p, mu, sigma, A, data):
    K = len(p) # states
    T = len(data)
    log_alpha = np.zeros((K, T))
    log_beta = np.zeros((K, T))

    log_condi_p = np.array([multivariate_normal.logpdf(data[0], mu[i], sigma[i]) for i in range(K)])
    log_alpha[:, 0] = np.log(p) + log_condi_p
    #print('t= 0 log_alpha = ', log_alpha[:,0])
    for t in range(1, T):
        log_prev_factor = log_matrix_dot(A, log_alpha[:, t-1])
        log_condi_p = (np.array([multivariate_normal.logpdf(data[t], mu[i], sigma[i]) for i in range(K)]))
        log_alpha[:, t] = log_condi_p +  log_prev_factor
        #print('t=', t, ' log_alpha = ', log_alpha[:,t])

    log_condi_p = (np.array([multivariate_normal.logpdf(data[-1], mu[i], sigma[i]) for i in range(K)]))
    log_beta[:,-1] = log_condi_p
    #print('t= 0 log_beta= ', log_beta[:,-1])
    for t in range(T-2, -1, -1):
        log_condi_p = (np.array([multivariate_normal.logpdf(data[t+1], mu[i], sigma[i]) for i in range(K)]))
        log_beta[:, t] = log_matrix_dot(A, log_beta[:, t+1]+log_condi_p)
        #print('t=', t, ' log_beta= ', log_beta[:,t])

    # gamma represents the probability of a latent variable given all the
    # observations
    log_gamma = log_alpha + log_beta
    log_Z = [log_dot(np.ones(K), log_gamma[:,t]) for t in range(T)]
    for i in range(K):
        for t in range(T):
            log_gamma[i, t] -= log_Z[t]
    gamma = np.exp(log_gamma)

    log_chsi = np.zeros((K, K, T-1))
    for t in range(T-1):
        log_condi_p = (np.array([multivariate_normal.logpdf(data[t+1], mu[k], sigma[k]) for k in range(K)]))
        for i in range(K):
            for j in range(K):
                log_chsi[i, j, t] = log_alpha[i,t] + log_beta[j, t+1] + np.log(A[i,j]) + log_condi_p[j] - log_Z[t]
                #log_chsi[i, j, t] = log_alpha[i,t] + log_gamma[j, t+1] + np.log(A[i,j]) + log_condi_p[j] - log_alpha[j, t+1]
    chsi = np.exp(log_chsi)
    return gamma, chsi

# Most probable states
def viterbi(p, A, mu, sigma, data):
    K = len(p)
    T = len(data)
    path = {}
    # Initialization
    log_v = np.zeros((K, T))
    log_condi_p = np.array([multivariate_normal.logpdf(data[0], mu[i], sigma[i]) for i in range(K)])
    log_v[:, 0] = np.log(p) + log_condi_p
    for k in range(K):
        path[k] = [k]
    # Propagation of the algorithm
    for t in range(1, T):
        newpath = {}
        log_condi_p = (np.array([multivariate_normal.logpdf(data[t], mu[i], sigma[i]) for i in range(K)]))
        for k in range(K):
            # V(t,k) = max_x P(y_t | k) a(x,k) V(t-1, x)
            (log_v[k,t], state) = max((log_condi_p[k] + np.log(A[s,k]) + log_v[s, t-1], s) for s in range(K))
            newpath[k] = path[state]+[k]
        path = newpath
    (_, state) = max((log_v[k, -1], k) for k in range(K))
    return np.array(path[state])

# Maximization step
def m_step(gamma, chsi, data):
    K, T = gamma.shape
    p = gamma[:, 0]
    A = np.sum(chsi, axis = 2)/np.sum(np.sum(chsi, axis=1), axis=1)
    mu = [sum(gamma[i, t]*data[t] for t in range(T))/np.sum(gamma[i,:]) for i in range(K)]
    sigma = [sum(gamma[i, t]*np.outer(data[t] - mu[i], data[t] - mu[i]) for t in range(T))/np.sum(gamma[i, :]) for i in range(K)]
    return p, A, mu, sigma

# Computes the log likelihood of the current parameters
def loglike(p, A, mu, sigma, data):
    gamma, chsi = forward_backward(p, mu, sigma, A, data)
    res = sum(sum(gamma[i,t]*multivariate_normal.logpdf(data[t], mu[i], sigma[i]) for i in range(K)) for t in range(T)) +sum(sum(sum(chsi[i,j,t]*np.log(A[i,j]) for i in range(K)) for j in range(K)) for t in range(T-1))
    return res/len(data)


def plot_gamma(gamma, N=100):
    plt.figure(1)
    plt.title('Probability of being in stage $k$ knowing all observations')
    plt.subplot(411)
    plt.plot(np.arange(N), gamma[0, :N])
    plt.ylabel('$p(q_1 | u_{1:T})$')
    plt.xlabel('iter')

    plt.subplot(412)
    plt.plot(np.arange(N), gamma[1, :N])
    plt.ylabel('$p(q_2 | u_{1:T})$')
    plt.xlabel('iter')

    plt.subplot(413)
    plt.plot(np.arange(N), gamma[2, :N])
    plt.ylabel('$p(q_3 | u_{1:T})$')
    plt.xlabel('iter')

    plt.subplot(414)
    plt.plot(np.arange(N), gamma[3, :N])
    plt.ylabel('$p(q_4 | u_{1:T})$')
    plt.xlabel('iter')

    plt.show()

def plot_loglik(loglik_data, loglik_test):
    N = len(loglik_test)
    plt.figure(3)
    plt.subplot(211)
    plt.plot(np.arange(N), loglik_data)
    plt.ylabel('$\ell(q| \theta)$')
    plt.title('Log-likelihood on train set')

    plt.subplot(212)
    plt.plot(np.arange(N), loglik_test)
    plt.ylabel('$\ell(q|\theta)$')
    plt.xlabel('iter')
    plt.title('Log-likelihood on test set')

    plt.show()


def compare_states(viterbi_prediction_train, marginal_prediction_train, viterbi_prediction_test, marginal_prediction_test, N=100):
    plt.figure(4)
    plt.subplot(211)
    plt.plot(np.arange(N), viterbi_prediction_train[:N], 'ro', label='Viterbi')
    plt.plot(np.arange(N), marginal_prediction_train[:N], 'bo', label='Marginal')
    plt.xlabel('Train set')
    plt.legend()
    plt.ylabel('Most probable state')

    plt.subplot(212)
    plt.plot(np.arange(N), viterbi_prediction_test[:N], 'ro', label='Viterbi')
    plt.plot(np.arange(N), marginal_prediction_test[:N], 'bo', label='Marginal')
    plt.xlabel('Test set')
    plt.ylabel('Most probable state')
    plt.show()


    return

if __name__ == '__main__':
    plt.style.use('ggplot')
    np.random.seed(0)
    np.set_printoptions(precision=2, suppress=True)
    data, rien = importdata('EMGaussian.data')
    test, rien = importdata('EMGaussian.test')
    T = len(data)
    K = 4
    N = 100

    # Transition matrix
    A = np.array([[1/2, 1/6, 1/6, 1/6], [1/6, 1/2, 1/6, 1/6], [1/6, 1/6, 1/2, 1/6], [1/6, 1/6, 1/6, 1/2]])

    # Initialization with EM on GMM
    p, mu, sigma = em.EM(K, 10, data)
    p = np.ones(K)/K
    print('Initial parameters')
    print('p=', p)
    print('A=', A)
    print('mu=', mu)
    print('sigma=', sigma)

    # EM iterations
    n_iter = 10
    loglik_data = np.zeros(n_iter)
    loglik_test = np.zeros(n_iter)
    for k in range(n_iter):
        print('EM-HMM iter=', k+1, '/', n_iter)
        # E-step
        gamma, chsi = forward_backward(p, mu, sigma, A, data)
        # M-step
        p, A, mu, sigma = m_step(gamma, chsi, data)
        # Stores likelihood
        loglik_data[k] = loglike(p, A, mu, sigma, data)
        loglik_test[k] = loglike(p, A, mu, sigma, test)

    mu = np.array(mu)
    print('p=', p)
    print('A=', A)
    print('mu=', mu)
    print('sigma=', sigma)
    mps = viterbi(p, A, mu, sigma, data)
    print('shape', mps.shape)
    print(mps[:N])
    clusters_marg = np.argmax(gamma, axis=0)
    print(clusters_marg[:N])
    clusters = mps

    # Plot clusters
    km.plot_clusters(clusters, mu, K, data, title='HMM clustering')

    mps_test = viterbi(np.ones(4)/4, A, mu, sigma, test)
    km.plot_clusters(mps_test, mu, K, test, title='Viterbi on test set')

    # Plot marginal probability on train set
    plot_gamma(gamma, N)

    # Plot marginal probability on test set
    gamma_test, chsi_test = forward_backward(p, mu, sigma, A, test)

    plot_gamma(gamma_test, N)
    # Plot log-likelihood
    plot_loglik(loglik_data, loglik_test)



    vit_pred_train = viterbi(p, A, mu, sigma, data)
    mar_pred_train = np.argmax(gamma, axis=0)

    vit_pred_test = viterbi(p, A, mu, sigma, test)
    mar_pred_test = np.argmax(gamma_test, axis=0)
    # Plot most probable states comparison
    compare_states(vit_pred_train, mar_pred_train, vit_pred_test, mar_pred_test, N)
