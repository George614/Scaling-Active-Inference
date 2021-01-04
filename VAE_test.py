import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import VAE_models

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
vae = VAE_models.VAE(latent_dim, kl_weight=kl_weight)

for epoch in range(epochs):
    print("=================== Epoch {} ==================".format(epoch+1))
    loss_accum = 0
    for step, x_train_batch in enumerate(train_dataset):
        loss = VAE_models.train_step(x_train_batch, vae, opt)
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