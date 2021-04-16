import tensorflow as tf
import numpy as np
import math


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                               tf.TensorSpec(shape=None, dtype=tf.float32)])
def squared_error_vec(x_reconst, x_true):
    """ Squared error dimension-wise """
    ############################################################################
    # Alex-style -- avoid complex operators like pow(.) whenever you can, below
    #               is the same as above
    ############################################################################
    diff = x_reconst - x_true
    se = diff * diff # squared error
    ############################################################################
    return se


@tf.function
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


@tf.function
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


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                               tf.TensorSpec(shape=None, dtype=tf.float32)])
def mse_with_sum(x_reconst, x_true):
    ''' Mean Squared Error with sum over last dimension '''
    sse = tf.math.reduce_sum(tf.math.square(x_reconst - x_true), axis=-1)
    return tf.math.reduce_mean(sse)


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                               tf.TensorSpec(shape=None, dtype=tf.float32)])
def mse(x_reconst, x_true):
    ''' Mean Squared Error '''
    #se = tf.math.square(x_reconst - x_true)
    return tf.math.reduce_mean(squared_error_vec(x_reconst,x_true))


@tf.function
def kl_d_old(mu_p, std_p, mu_q, std_q, keep_batch=False):
    ''' KL-Divergence function for 2 diagonal Gaussian distributions '''
    # reference: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    EPSILON = 1e-5
    k = tf.cast(tf.shape(mu_p)[-1], tf.float32)
    log_var = tf.math.log(tf.reduce_prod(tf.math.square(std_q), axis=-1)/tf.reduce_prod(tf.math.square(std_p), axis=-1) + EPSILON)
    mu_var_multip = tf.math.reduce_sum((mu_p-mu_q) / (tf.math.square(std_q) + EPSILON) * (mu_p-mu_q), axis=-1)
    trace = tf.math.reduce_sum(tf.math.square(std_p) / (tf.math.square(std_q) + EPSILON), axis=-1)
    kld = 0.5 * (log_var - k + mu_var_multip + trace)
    if not keep_batch:
        kld = tf.math.reduce_mean(kld)
    return kld


@tf.function
def g_nll_old(mu, std, x_true, keep_batch=False):
    ''' Gaussian Negative Log Likelihood loss function '''
    EPSILON = 1e-5
    nll = 0.5 * tf.math.log(2 * math.pi * tf.math.square(std)) + tf.math.square(x_true - mu) / (2 * tf.math.square(std) + EPSILON)
    nll = tf.reduce_sum(nll, axis=-1)
    if not keep_batch:
        nll = tf.reduce_mean(nll)
    return nll


@tf.function
def huber(y_true, y_pred, delta=1.0, keep_batch=False):
    error = y_true - y_pred
    within_d = tf.math.less_equal(tf.abs(error), delta)
    within_d = tf.cast(within_d, dtype=tf.float32)
    loss_in = 0.5 * error * error
    loss_out = 0.5 * delta * delta + delta * (tf.abs(error) - delta)
    loss = within_d * loss_in + (1 - within_d) * loss_out
    if keep_batch:
        return loss
    return tf.math.reduce_mean(loss, axis=0)