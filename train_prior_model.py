import logging
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from datetime import datetime
from utils import PARSER
from AI_model import Encoder, g_nll

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

@tf.function
def train_step(model, optimizer, o_cur):
    with tf.GradientTape() as tape:
        z_sample, mu_o, std_o = model(o_cur)
        gnll = g_nll(mu_o, std_o, o_cur)

    gradients = tape.gradient(gnll, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return gnll


if __name__ == "__main__":
    print("Prior model training data path: ", args.prior_data_path)
    all_data = np.load(args.prior_data_path + "/all_human_data.npy", allow_pickle=True)
    
    # one-hot encoded the action vector
    all_data = all_data[1:-1, 1:, :]  # exclude samples with imcomplete sequence
    all_actions = all_data[:, :, 2].reshape(-1)
    all_actions_onehot = tf.keras.utils.to_categorical(all_actions)
    all_obv = all_data[:, :, 0:2].reshape(-1, 2)
    all_data = np.hstack((all_obv, all_actions_onehot))
    # all_data = all_obv
    mask = np.not_equal(all_data[:, 0], 0)
    all_data = all_data[mask]
    
    o_size = 2
    batch_size = 128
    n_epoch = 100
    train_dataset = tf.data.Dataset.from_tensor_slices(all_data)
    train_dataset = train_dataset.shuffle(buffer_size=args.buffer_size).batch(batch_size)
    model_save_path = "results/prior_model/{}/".format(args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # use tensorboard for monitoring training if needed
    now = datetime.now()
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard', now.strftime("%b-%d-%Y %H-%M-%S"))
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    opt = tf.keras.optimizers.get(args.vae_optimizer)
    opt.__setattr__('learning_rate', args.vae_learning_rate)

    # initial our model using parameters in the config file
    prior_model = Encoder(o_size, o_size, name='prior_model')

    total_train_steps = 0
    
    for epoch in range(n_epoch):
        print("=================== Epoch {} ==================".format(epoch+1))
        for training_step, x_batch in enumerate(train_dataset):
            # split the batch data into observation and action (disgard reward for now)
            o_cur_batch, a_prev_batch = x_batch[:, 0:2], x_batch[:, 2:]
            gnll = train_step(prior_model, opt, o_cur_batch)

            total_train_steps += 1
            tf.summary.scalar('GNLL', gnll.numpy(), step=total_train_steps)
            if training_step % 100 == 0:
                print("training_step {}, gnll: {:.3f}".format(training_step+1, gnll.numpy()))

    # save the prior model
    tf.saved_model.save(prior_model, model_save_path)
    print("> Trained the prior model. Saved in:", model_save_path)