import tensorflow as tf
import math


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='mean'),
                              tf.TensorSpec(shape=None, dtype=tf.float32, name='std')])
def sample(mu, std):
    # reparameterization trick
    z_sample = mu + std * tf.random.normal(tf.shape(std))
    return z_sample


@tf.function
def sample_k_times(k, mu, std):
    multiples = tf.constant((1, k, 1), dtype=tf.int32)
    # shape of mu_, std_ and z_sample: (batch x n_samples x vec_dim)
    mu_ = tf.tile(tf.expand_dims(mu, axis=1), multiples)
    std_ = tf.tile(tf.expand_dims(std, axis=1), multiples)
    # reparameterization trick
    z_sample = mu_ + std_ * tf.random.normal(tf.shape(std_))
    return z_sample


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def swish(x):
    ''' swish activation function '''
    return x * tf.math.sigmoid(x)


class Dense(tf.Module):
    ''' fully-connected NN layer written in a TF2 way '''
    def __init__(self, in_features, out_features, name=None, trainable=True):
        super().__init__(name=name)
        bound = 1 / math.sqrt(in_features)
        self.W = tf.random.uniform((in_features, out_features), -bound, bound)
        self.W = tf.Variable(self.W, name='W', trainable=trainable)
        self.b = tf.zeros((out_features,))
        self.b = tf.Variable(self.b, name='b', trainable=trainable)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b


class Encoder(tf.Module):
    ''' transition / posterior model of the active inference framework '''
    def __init__(self, dim_input, dim_z, n_samples=1, name='Encoder', activation='relu'):
        super().__init__(name=name)
        self.dim_z = dim_z  # latent dimension
        self.N = n_samples
        self.dense_input = Dense(dim_input, 64, name='dense_input')
        self.dense_e1 = Dense(64, 64, name='dense_1')
        self.dense_mu = Dense(64, dim_z, name='dense_mu')
        self.dense_raw_std = Dense(64, dim_z, name='dense_raw_std')
        self.mu = None
        self.std = None
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'relu6':
            self.activation = tf.nn.relu6
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, X):
        z = self.dense_input(X)
        z = self.activation(z)
        z = self.dense_e1(z)
        z = self.activation(z)
        mu = self.dense_mu(z)
        raw_std = self.dense_raw_std(z)
        # softplus is supposed to avoid numerical overflow
        std = tf.clip_by_value(tf.math.softplus(raw_std), 0.01, 10.0)
        if self.mu is None:
            batch_size = tf.shape(X)[0]
            self.mu = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
            self.std = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
        else:
            self.mu.assign(mu)
            self.std.assign(std)
        z_sample = sample(mu, std)

        return z_sample, mu, std


class FlexibleEncoder(tf.Module):
    ''' Generic Gaussian encoder model based on Dense layer. Output mean and std as well
    as samples from the learned distribution '''
    def __init__(self, layer_dims, n_samples=1, name='Encoder', activation='relu'):
        super().__init__(name=name)
        self.dim_z = layer_dims[-1]
        self.N = n_samples
        self.layers =[]
        for i in range(len(layer_dims)-1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1]))
        # add another group of neurons for mu/std in the last layer
        self.layers.append(Dense(layer_dims[-2], self.dim_z))
        self.mu = None
        self.std = None
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'relu6':
            self.activation = tf.nn.relu6
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):
        for layer in self.layers[:-2]:
            x = layer(x)
            x = self.activation(x)
        mu = self.layers[-2](x)
        raw_std = self.layers[-1](x)
        # softplus is supposed to avoid numerical overflow
        std = tf.clip_by_value(tf.math.softplus(raw_std), 0.01, 10.0)
        # log_sigma = self.layers[-1](x)
        # std = tf.math.exp(log_sigma)
        # std = tf.clip_by_value(std, 0.01, 10.0)
        if self.mu is None:
            batch_size = tf.shape(x)[0]
            self.mu = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
            self.std = tf.Variable(tf.zeros((batch_size, self.dim_z)), trainable=False)
        else:
            self.mu.assign(mu)
            self.std.assign(std)
        z_sample = sample(mu, std)

        return z_sample, mu, std #, log_sigma


class FlexibleMLP(tf.Module):
    ''' Simple multi-layer perceptron model taking layer parameters as input '''
    def __init__(self, layer_dims, name='MLP', activation='relu6', trainable=True): # relu6, or tanh
        super().__init__(name=name)
        self.layers =[]
        for i in range(len(layer_dims)-1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1], trainable=trainable))
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu6':
            self.activation = tf.nn.relu6
        elif activation == 'swish':
            self.activation = swish
        else:
            raise ValueError('incorrect activation type')

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)  # linear activation for the last layer
        return x