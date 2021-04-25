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


@tf.function
def drop_out(input, rate=0.0, seed=69):
    """
        Custom drop-out function -- returns output as well as binary mask
        @author Alex Ororbia
    """
    mask = tf.math.less_equal( tf.random.uniform(shape=(input.shape[0],input.shape[1]), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),(1.0 - rate))
    mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
    output = input * mask
    return output, mask


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
        z_l = tf.matmul(X, self.W) + self.b
        return z_l


class LayerNorm(tf.Module):
    ''' 
    Layer normalization NN layer
    @author Alexander Ororbia
    '''
    def __init__(self, in_features, out_features, name=None, trainable=True):
        super().__init__(name=name)
        self.var_eps = 1e-12
        self.alpha = tf.Variable(tf.zeros([1,out_features]), trainable=trainable)
        self.beta = tf.Variable(tf.zeros([1,out_features]), trainable=trainable)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, X):
        X_ = X
        # apply standardization based on layer normalization
        u = tf.reduce_mean(X_, keepdims=True)
        s = tf.reduce_mean(tf.pow(X_ - u, 2), axis=-1, keepdims=True)
        X_ = (X_ - u) / tf.sqrt(s + self.var_eps)
        # apply layer normalization re-scaling
        X_ = tf.multiply(self.alpha, X_) + self.beta # same as Hadamard product
        return X_


class LayerNormalization(tf.Module):
    ''' 
    Layer normalization layer 
    ref. https://arxiv.org/pdf/1607.06450.pdf
    '''
    def __init__(self, shape, gamma=True, beta=True, epsilon=1e-10, name="LayerNorm"):
        super().__init__(name=name)
        if isinstance(shape, int):
            normal_shape = (shape,)
        else:
            normal_shape = (shape[-1],)
        self.normal_shape =normal_shape 
        self.epsilon = epsilon
        if gamma:
            self.gamma = tf.ones(self.normal_shape, name="gamma")
            self.gamma = tf.Variable(self.gamma, trainable=True)
        if beta:
            self.beta = tf.zeros(self.normal_shape, name="beta")
            self.beta = tf.Variable(self.beta, trainable=True)

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.assign(tf.ones(self.normal_shape))
        if self.beta is not None:
            self.beta.assign(tf.zeros(self.normal_shape))

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, X):
        mu = tf.math.reduce_mean(X, axis=-1, keepdims=True)
        var = tf.math.reduce_mean((X - mu) * (X - mu), axis=-1, keepdims=True)
        std = tf.math.sqrt(var + self.epsilon)
        y = (X - mu) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y


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
    def __init__(self, layer_dims, n_samples=1, name='Encoder', activation='relu', layer_norm=False, dropout_rate=0.0):
        super().__init__(name=name)
        self.dim_z = layer_dims[-1]
        self.N = n_samples
        self.layers =[]
        for i in range(len(layer_dims) - 1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1]))
        # add another group of neurons for mu/std in the last layer
        self.layers.append(Dense(layer_dims[-2], self.dim_z))
        self.mu = None
        self.std = None
        self.layer_norm = tf.Variable(layer_norm, trainable=False)
        self.dropout_rate = tf.Variable(dropout_rate, trainable=False)
        self.norm_layers = None
        if tf.equal(self.layer_norm, True):
            self.norm_layers = []
            for i in range(len(layer_dims) - 2):
                self.norm_layers.append(LayerNormalization(layer_dims[i+1]))
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
        for i in range(len(self.layers) - 2):
            x = self.layers[i](x)
            if self.norm_layers is not None:
                x = self.norm_layers[i](x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = tf.nn.dropout(x, rate=self.dropout_rate)
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
    def __init__(self, layer_dims, name='MLP', activation='relu6', trainable=True, layer_norm=False):
        super().__init__(name=name)
        self.layers =[]
        for i in range(len(layer_dims)-1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1], trainable=trainable))
        self.layer_norm = tf.Variable(layer_norm, trainable=False)
        if self.layer_norm:
            self.norm_layers = []
            for i in range(len(layer_dims) - 2):
                self.norm_layers.append(LayerNormalization(layer_dims[i+1]))
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
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.layer_norm:
                x = self.norm_layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)  # linear activation for the last layer
        return x
