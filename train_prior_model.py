import logging
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from datetime import datetime
from utils import PARSER
from AI_model import Encoder, g_nll_old

AUTOTUNE = tf.data.experimental.AUTOTUNE
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

    
@tf.function
def train_step(model, optimizer, o_cur, o_next):
    with tf.GradientTape() as tape:
        z_sample, mu_o, std_o = model(o_cur)
        gnll = g_nll_old(mu_o, std_o, o_next)

    gradients = tape.gradient(gnll, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return gnll


@tf.function
def eval_step(model, o_cur, o_next):
    z_sample, mu_o, std_o = model(o_cur)
    gnll = g_nll_old(mu_o, std_o, o_next)
    return gnll
    

if __name__ == "__main__":
    # print("Prior model training data path: ", args.prior_data_path)
    # all_data = np.load(args.prior_data_path + "/all_human_data.npy", allow_pickle=True)
    # all_data = all_data[1:-1, 1:, :]  # exclude samples with imcomplete sequence

    # obv_all = []
    # for i in range(len(all_data)):
    #     o_episode = all_data[i, :, :2]
    #     mask = np.not_equal(o_episode[:, 0], 0)
    #     o_episode = o_episode[mask]
    #     o_t = o_episode[:-1, :]
    #     o_tp1 = o_episode[1:, :]
    #     obv_all.append(np.hstack((o_t, o_tp1)))
    
    # obv_all = np.vstack(obv_all)

    print("Prior model training data path: ", args.zoo_data_path)
    all_data = np.load(args.zoo_data_path + "/zoo-agent-mcar.npy", allow_pickle=True)
    obv_t, obv_tp1, done = all_data[:, :2], all_data[:, 4:6], all_data[:, 6]
    obv_all = np.hstack((obv_t, obv_tp1))
    obv_all = obv_all[np.not_equal(done, 1)]
    
    o_size = 2
    batch_size = 128
    n_epoch = 100
    test_n_samples = len(obv_all) // 10  # use 10% of data for testing
    all_dataset = tf.data.Dataset.from_tensor_slices(obv_all)
    test_dataset = all_dataset.take(test_n_samples)
    test_dataset = test_dataset.batch(test_n_samples)
    train_dataset = all_dataset.skip(test_n_samples)
    train_dataset = train_dataset.shuffle(buffer_size=args.buffer_size).batch(batch_size)
    model_save_path = "results/prior_model/{}/".format(args.env_name)
    model_save_path = os.path.join(model_save_path, "zoo_data")
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
            o_cur_batch, o_next_batch = x_batch[:, 0:2], x_batch[:, 2:]
            gnll = train_step(prior_model, opt, o_cur_batch, o_next_batch)

            total_train_steps += 1
            tf.summary.scalar('GNLL', gnll.numpy(), step=total_train_steps)
            if training_step % 100 == 0:
                print("training_step {}, gnll: {:.3f}".format(training_step+1, gnll.numpy()))
        
        # evaluate the network
        for sample in test_dataset:
            o_cur_batch, o_next_batch = sample[:, 0:2], sample[:, 2:]
            gnll_test = eval_step(prior_model, o_cur_batch, o_next_batch)
            tf.summary.scalar('GNLL_Test', gnll_test.numpy(), step=total_train_steps)
            print("epoch {}, gnll_test: {:.3f}".format(epoch+1, gnll_test.numpy()))
    
    # save the prior model
    tf.saved_model.save(prior_model, model_save_path)
    print("> Trained the prior model. Saved in:", model_save_path)