import tensorflow as tf
import numpy as np
import sys

"""
Diagnostic/Sanity-Checking Program:
Code that proves the derived formulas for NLL and KL of a multivariate Gaussian
with strictly assumed diagonal covariances match the numerically
problematic formulas for NLL and KL that use the formulation for a full covariance
multivariate Gaussian but ultimately applied to diagonal covariance, i.e., diag(Sigma).

@author Alexander G. Ororbia II
"""

seed = 69
#tf.random.set_random_seed(seed=seed)
tf.random.set_seed(seed=seed)
np.random.seed(seed)

def sample_uniform(n_s, n_dim, minv=0., maxv=1.):
    eps = tf.random.uniform(shape=(n_s,n_dim), minval=minv, maxval=maxv, dtype=tf.float32, seed=69)
    return eps

def sample_gaussian(n_s, mu=0.0, sig=1.0, n_dim=-1):
    """
        Samples a multivariate Gaussian assuming at worst a diagonal covariance
    """
    dim = n_dim
    if dim <= 0:
        dim = mu.shape[1]
    eps = tf.random.normal([n_s, dim], mean=0.0, stddev=1.0, seed=69)
    return mu + eps * sig

def kl_d(mu_p, sigSqr_p, log_sig_p, mu_q, sigSqr_q, log_sig_q, keep_batch=False):
    """
         Kullback-Leibler (KL) Divergence function for 2 multivariate Gaussian distributions
         with strict diagonal covariances.

         Follows formula, where p(x) = N(x ; mu_p,sig^2_p) and q(x) = N(x ; mu_q,sig^2_q):
         KL(p||q) = log(sig_q/sig_p) + (sig^2_p + (mu_p - mu_q)^2)/(2 * sig^2_q) - 1/2
                  = [ log(sig_q) - log(sig_p) ] + (sig^2_p + (mu_p - mu_q)^2)/(2 * sig^2_q) - 1/2
    """
    ############################################################################
    # Alex-style - avoid logarithms whenever you can pre-compute them
    # I like the variance-form of G-KL, I find it generally to be more stable
    # Note that I expanded the formula a bit further using log difference rule
    ############################################################################
    eps = 1e-6
    term1 = log_sig_q - log_sig_p
    diff = mu_p - mu_q
    term2 = (sigSqr_p + (diff * diff))/(sigSqr_q * 2 + eps)
    KLD = term1 + term2 - 1/2
    KLD = tf.reduce_sum(KLD, axis=-1) #gets KL per sample in batch (a column vector)
    ############################################################################
    if not keep_batch:
        KLD = tf.math.reduce_mean(KLD)
    return KLD

def g_nll(X, mu, sigSqr, log_sig=None, keep_batch=False):
    """ Gaussian Negative Log Likelihood loss function
        --> assumes N(x; mu, sig^2)

        mu <- external mean
        sigSqr <- external variance
        log_sig <- pre-computed log(sigma) (numerically stable form)
        keep_batch <-
    """
    ############################################################################
    # Alex-style - avoid logarithms whenever you can pre-compute them
    # I like the variance-form of GNLL, I find it generally to be more stable
    eps = 1e-6
    diff = X - mu # pre-compute this quantity
    term1 = -( (diff * diff)/(sigSqr *2 + eps) ) # central term
    if log_sig is not None:
        # expanded out the log(sigma * sigma) = log(sigma) + log(sigma)
        term2 = -(log_sig + log_sig) * 0.5 # numerically more stable form
    else:
        term2 = -tf.math.log(sigSqr) * 0.5
    term3 = -tf.math.log(np.pi * 2) * 0.5 # constant term
    nll = -( term1 + term2 + term3 ) # -( LL ) = NLL
    nll = tf.reduce_sum(nll, axis=-1) # gets GNLL per sample in batch (a column vector)
    ############################################################################
    if not keep_batch:
        nll = tf.reduce_mean(nll)
    return nll


def kl_d_old(mu_p, std_p, mu_q, std_q, keep_batch=False):
    ''' KL-Divergence function for 2 diagonal Gaussian distributions '''
    EPSILON = 1e-6
    # reference: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    k = tf.cast(tf.shape(mu_p)[-1], tf.float32)
    log_var = tf.math.log(tf.reduce_prod(tf.math.square(std_q), axis=-1)/tf.reduce_prod(tf.math.square(std_p), axis=-1) + EPSILON)
    mu_var_multip = tf.math.reduce_sum((mu_p-mu_q) / (tf.math.square(std_q) + EPSILON) * (mu_p-mu_q), axis=-1)
    trace = tf.math.reduce_sum(tf.math.square(std_p) / (tf.math.square(std_q) + EPSILON), axis=-1)
    kld = 0.5 * (log_var - k + mu_var_multip + trace)
    if not keep_batch:
        kld = tf.math.reduce_mean(kld)
    return kld


def g_nll_old(x_true, mu, std, keep_batch=False):
    ''' Gaussian Negative Log Likelihood loss function '''
    EPSILON = 1e-6
    nll = 0.5 * tf.math.log(2 * np.pi * tf.math.square(std)) + tf.math.square(x_true - mu) / (2 * tf.math.square(std) + EPSILON)
    nll = tf.reduce_sum(nll, axis=-1)
    if not keep_batch:
        nll = tf.reduce_mean(nll)
    return nll

# create synthetic datasets P and Q
n_v = 8 # number of dimensions in input space
dataset_size = 50 # number data points

mu_ = sample_gaussian(n_s=1, mu=0.0, sig=0.2, n_dim=n_v)
sd_ = sample_uniform(n_s=1, n_dim=n_v, maxv=2)
P = sample_gaussian(n_s=dataset_size, mu=mu_, sig=sd_, n_dim=n_v)
# compute MLE of mu and sigma to simulate an external model
mu_p = tf.reduce_mean(P,axis=0,keepdims=True)
diff = P - mu_p
Sigma_p = tf.matmul(diff, diff, transpose_a=True)/((P.shape[0] - 1) * 1.0)
sigSqr_p = tf.expand_dims(tf.linalg.diag_part(Sigma_p),axis=0) #tf.reduce_sum(diff * diff,axis=0,keepdims=True)/((X.shape[0] - 1) * 1.0)
sig_p = tf.math.sqrt(sigSqr_p)

mu_ = sample_gaussian(n_s=1, mu=0.1, sig=0.3, n_dim=n_v)
sd_ = sample_uniform(n_s=1, n_dim=n_v, maxv=3)
Q = sample_gaussian(n_s=dataset_size, mu=mu_, sig=sd_, n_dim=n_v)
mu_q = tf.reduce_mean(Q,axis=0,keepdims=True)
diff = Q - mu_q
Sigma_q = tf.matmul(diff, diff, transpose_a=True)/((Q.shape[0] - 1) * 1.0)
sigSqr_q = tf.expand_dims(tf.linalg.diag_part(Sigma_q),axis=0)
sig_q = tf.math.sqrt(sigSqr_q)

# Alex-style formulas
print("----------------------------------")
print("---- Alex-Style Formulas ----")
nll = g_nll(X=P, mu=mu_p, sigSqr=sigSqr_p, log_sig=tf.math.log(sig_p))
print("p.NLL = ",nll)
nll = g_nll(X=Q, mu=mu_q, sigSqr=sigSqr_q, log_sig=tf.math.log(sig_q))
print("q.NLL = ",nll)

#kl = kl_d(mu_p, sigSqr_p, tf.math.log(sig_p), mu_p, sigSqr_p, tf.math.log(sig_p))
#print("KL(p,p) = ",kl)
#kl = kl_d(mu_q, sigSqr_q, tf.math.log(sig_q), mu_q, sigSqr_q, tf.math.log(sig_q))
#print("KL(q,q) = ",kl)
kl = kl_d(mu_p, sigSqr_p, tf.math.log(sig_p), mu_q, sigSqr_q, tf.math.log(sig_q))
print("KL(p,q) = ",kl)

print("----------------------------------")
print("---- Standard Formulas ----")
nll = g_nll_old(x_true=P, mu=mu_p, std=sig_p)
print("p.NLL = ",nll)
nll = g_nll_old(x_true=Q, mu=mu_q, std=sig_q)
print("q.NLL = ",nll)

kl = kl_d_old(mu_p, sig_p, mu_q, sig_q)
print("KL(p,q) = ",kl)
