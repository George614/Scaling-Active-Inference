# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:33:04 2020

Simple Variational Autoencoder based on TF 2

@author: George (Zhizhuo) Yang
"""
import numpy as np
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

""" try to write DNN and VAE in TF2 based on a numpy implementation
class Layer(object):
    def set_input_shape(self, shape):
        ''' 
        sets the shape that the layer expects of the input in the
        forward pass
        '''
        self.input_shape = shape
        
    def layer_name(self):
        ''' name of the layer, used in model summary '''
        return self.__class__.__name__
    
    def parameters(self):
        ''' number of trainable parameters of the layer '''
        return 0
        
    def forward(self, X_in, training):
        ''' forward propogation for the network/layer '''
        raise NotImplementedError()
        
    def backward(self):
        ''' propogates the accumulated gradient backwards in the network '''
        raise NotImplementedError()
        
    def output_shape(self):
        '''the shape of the output produced by the forward pass '''
        raise NotImplementedError()
        

class Dense(Layer):
    ''' fully-connected NN layer '''
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.b = None
    
    def initialize(self):
        ''' initialize the weights '''
        bound = 1 / math.sqrt(self.input_shape[0])
        self.W = tf.random_uniform((self.input_shape[0], self.n_units), -bound, bound)
        self.W = tf.Variable(self.W)
        self.b = tf.zeros((1, self.n_units))
        self.b = tf.Variable(self.b)
        
    def parameters(self):
        return tf.math.reduce_prod(self.W.shape) + tf.math.reduce_prod(self.b.shape)

    def forward(self, X_in, training=True):
        self.layer_input = X_in
        return tf.matmul(X_in, self.W) + self.b
    
    def backward(self, tape, loss)
        
"""
        

def sigmoid_cross_entropy_with_logits(x, y):
    ''' logistic loss with sigmoid built-in.
        x: logits
        y: lables
    '''
    return y * -tf.math.log(tf.math.sigmoid(x)) + (1-y) * -tf.math.log(1 - tf.math.sigmoid(x))


class Dense(tf.Module):
    ''' fully-connected NN layer written in a TF2 way '''
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        bound = 1 / math.sqrt(in_features)
        self.W = tf.random_uniform((in_features, out_features), -bound, bound)
        self.W = tf.Variable(self.W, name='W')
        self.b = tf.zeros((out_features,))
        self.b = tf.Variable(self.b, name='b')
    
    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b
        
    
class FlexibleDense(tf.Module):
    # No need to specify `in_features`
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features
    
    @tf.Module.with_name_scope # this can group operations in TensorBoard
    def __call__(self, X):
        # Create variables on first call.
        if not self.is_built:
          self.W = tf.Variable(
            tf.random.normal([X.shape[-1], self.out_features]), name='W')
          self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
          self.is_built = True
        return tf.matmul(X, self.W) + self.b


class Sampler(tf.Module):
    ''' sampler for latent space using the reparameterization trick '''
    def __init__(self, name='sampler'):
        super().__init__(name=name)
    
    def __call__(self, inputs):
        mu, raw_std = inputs
        # softplus is supposed to avoid numerical overflow
        # 1e-18 is tested for the squared std (used later in loss) i.e.
        # tf.math.log(1e-36) is legal
        std = tf.math.softplus(raw_std) + 1e-18
        # read dims individually since the tf.shape() doesn't return a list
        # but returns a Tensor instead (which cannot be interated through)
        batch_size = tf.shape(mu)[0]
        dim_z = tf.shape(mu)[1]
        # reparameterization trick
        z_sample = mu + std * tf.random.normal((batch_size, dim_z))
        return z_sample, std


class Encoder(tf.Module):
    ''' Encoder part of VAE '''
    def __init__(self, dim_z, name='encoder', activation='leaky_relu'):
        super().__init__(name=name)
        self.dim_z = dim_z  # latent dimension
        self.dense_input = FlexibleDense(256, name='dense_input')
        self.dense_e1 = FlexibleDense(128, name='dense_1')
        # self.dense_e2 = FlexibleDense(128, name='dense_2')
        self.dense_mu = FlexibleDense(dim_z, name='dense_mu')
        self.dense_raw_std = FlexibleDense(dim_z, name='dense_raw_std')
        self.sampler = Sampler()
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        else:
            raise ValueError('incorrect activation type')    
    
    @tf.Module.with_name_scope
    def __call__(self, X):
        z = self.dense_input(X)
        z = self.activation(z)
        z = self.dense_e1(z)
        z = self.activation(z)
        # z = self.dense_e2(z)
        # z = self.activation(z)
        mu = self.dense_mu(z)
        raw_std = self.dense_raw_std(z)
        z_sample, std = self.sampler((mu, raw_std))
        return z_sample, mu, std


class Decoder(tf.Module):
    ''' Decoder part of VAE '''
    def __init__(self, dim_x, name='decoder', activation='leaky_relu'):
        super().__init__(name=name)
        self.dim_x = dim_x
        self.dense_z_input = FlexibleDense(128, name='dense_z_input')
        self.dense_d1 = FlexibleDense(256, name='dense_d1')
        # 2 layers before the output layer seem to be enough
        # self.dense_d2 = FlexibleDense(512, name='dense_d2')
        self.dense_img = FlexibleDense(self.dim_x, name='dense_x_output')
        # tanh doesn't work (always generates exactly same data) since the 
        # latent variables tend to be all the same for intances in a batch
        # after several (>=3) rounds of activations
        if activation == 'tanh': 
            self.activation = tf.nn.tanh
        elif activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        else:
            raise ValueError('incorrect activation type')
       
    @tf.Module.with_name_scope
    def __call__(self, z):
        x_output = self.dense_z_input(z)
        x_output = self.activation(x_output)
        x_output = self.dense_d1(x_output)
        x_output = self.activation(x_output)
        # x_output = self.dense_d2(x_output)
        # x_output = self.activation(x_output)
        x_output = self.dense_img(x_output)
        return x_output
    

class VAE(tf.Module):
    def __init__(self, dim_z, kl_weight=1, name='VAE'):
        super().__init__(name=name)
        self.img_dims = (28, 28)
        self.dim_x = tf.math.reduce_prod(self.img_dims)
        self.dim_z = dim_z  # latent dimension
        self.encoder = Encoder(self.dim_z)
        self.decoder = Decoder(self.dim_x)
        self.kl_weight = kl_weight
        
    @tf.Module.with_name_scope
    def neg_log_likelihood(self, x_reconst, x_true):
        # calculate cross-entropy for each pixel
        # cross_entropy = sigmoid_cross_entropy_with_logits(x_reconst, x_true)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconst, labels=x_true)
        # calculate negtive log-likelihood for one image
        nll = tf.math.reduce_sum(cross_entropy, axis=1)
        return tf.math.reduce_mean(nll)
    
    @tf.Module.with_name_scope
    def __call__(self, x_inputs):
        z_sample, mu, std = self.encoder(x_inputs)
        x_reconst = self.decoder(z_sample)
        # first calculate KL divergence for each image (each dim along z is
        # inside the tf.math.reduce_sum() )
        kl_d = -0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(std)) - tf.math.square(mu) - tf.math.square(std), axis=1)
        # then calculate the batch mean and scale it by the KL weight factor
        kl_d = self.kl_weight * tf.math.reduce_mean(kl_d)
        # negative log-likelihood as reconstruction loss
        nll = self.neg_log_likelihood(x_reconst, x_inputs)
        total_loss = nll + kl_d
        return x_reconst, total_loss
    

# use tf.function makes a graph (run in graph mode)
@tf.function  # use decorator to let TF2 make training performance improvement
def train_step(x_true, model, optimizer):
    with tf.GradientTape() as tape:
        x_reconst, loss = model(x_true)
    # since all layer/model classes are subclassed from tf.Modules, it's
    # eaiser to gather all trainable variables from all layers then calculate
    # gradients for all of them. Note that tf.Module is used instead of 
    # keras.Layer or keras.Model because tf.Module is cleaner / bare-bone
    gradients = tape.gradient(loss, model.trainable_variables)
    # use back-prop algorithms for optimization for now
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#%%
###################### experiment on MNIST ######################
# hyperparameter setting
train_size = 60000
test_size = 10000
batch_size = 64
latent_dim = 32
lr = 1e-3
kl_weight = 3
epochs = 300

# loading data using TF2 Dataset
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[:train_size]
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_on = x_train >= 0.5
x_off = x_train < 0.5
x_train_binary = x_train.copy()
x_train_binary[x_on] = 1.0
x_train_binary[x_off] = 0.0
train_dataset = tf.data.Dataset.from_tensor_slices(x_train_binary)
train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(batch_size)

# training the model
opt = tf.optimizers.Adam(learning_rate=lr)
vae = VAE(latent_dim, kl_weight=kl_weight)

for epoch in range(epochs):
    print("=================== Epoch {} ==================".format(epoch+1))
    loss_accum = 0
    for step, x_train_batch in enumerate(train_dataset):
        loss = train_step(x_train_batch, vae, opt)
        loss_accum += loss.numpy()
        if step % 100 == 0:
            print("step {}, loss: {:.4f}".format(step, loss_accum / (step+1)))
        
# save the trained model
path = os.getcwd()
model_path = os.path.join(path, 'VAE_trained')
tf.saved_model.save(vae, model_path)

#%% sampling from trained VAE     
num_gen_samples = 10
z_samples =  tf.random.normal((num_gen_samples, latent_dim))
x_generated = vae.decoder(z_samples)
x_generated = tf.math.sigmoid(x_generated)
x_gen_np = x_generated.numpy()
x_gen_np = x_gen_np.reshape((num_gen_samples, 28, 28))
for i in range(num_gen_samples):
    plt.imsave(path + '/generated_{:03d}.png'.format(i), x_gen_np[i])
    plt.imshow(x_gen_np[i])
    plt.show()
    