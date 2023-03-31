import numpy as np
import tensorflow as tf
from pinn.utils import connect_dist_grad, atomic_dress
from pinn.io import sparse_batch
from pinn.optimizers import get
from keras import backend as K
import time

def get_traintest_sets(dataset=None, buffer_size=20000, batch_size=None):
    train_set = dataset['train'].shuffle(buffer_size).apply(sparse_batch(batch_size))
    test_set = dataset['test'].apply(sparse_batch(batch_size))
    return (train_set, test_set, batch_size)


def preprocess_traintest_sets(train_set, test_set, network=None):
    for batch in train_set:
        batch = network.preprocess(batch)
        connect_dist_grad(batch)
        # print("Train set: Preprocessed ind_1 shape: ", batch['ind_1'].shape)
        # print("Train set: Preprocessed ind_2 shape: ", batch['ind_2'].shape)
        # print("Train set: Preprocessed elems shape: ", batch['elems'].shape)
        # print("Train set: Preprocessed coord shape: ", batch['coord'].shape)
        # print("Train set: Preprocessed e_data shape: ", batch['e_data'].shape)
        # print("Train set: Preprocessed f_data shape: ", batch['f_data'].shape)
    for batch in test_set:
        batch = network.preprocess(batch)
        connect_dist_grad(batch)
        # print("Test set: Preprocessed ind_1 shape: ", batch['ind_1'].shape)
        # print("Test set: Preprocessed ind_2 shape: ", batch['ind_2'].shape)
        # print("Test set: Preprocessed elems shape: ", batch['elems'].shape)
        # print("Test set: Preprocessed coord shape: ", batch['coord'].shape)
        # print("Test set: Preprocessed e_data shape: ", batch['e_data'].shape)
        # print("Test set: Preprocessed f_data shape: ", batch['f_data'].shape)



def train_and_evaluate_network(network=None, params=None, train_set=None, test_set=None, batch_size=256, epochs=1):



    # Instantiate an optimizer
    optimizer = get(params['optimizer'])
    # Define a loss function
    loss_fn = tf.keras.losses.mse
    # Define metrics
    train_loss_metric = tf.keras.metrics.MeanSquaredError()
    val_loss_metric = tf.keras.metrics.MeanSquaredError()
    train_err_metric = tf.keras.metrics.RootMeanSquaredError()
    val_err_metric = tf.keras.metrics.RootMeanSquaredError()
    

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time_epoch = time.time()
        hund_step_times = []

        # Iterate over the batches of the dataset.
        for step, batch in enumerate(train_set):
            
            

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                pred = network(batch, training=True)  # Logits for this minibatch

                ind = batch['ind_1']
                nbatch = tf.reduce_max(ind)+1
                pred = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)
                e_data = batch['e_data']

                if params['e_dress']:
                    e_data -= atomic_dress(batch, params['e_dress'], dtype=pred.dtype)
                    e_data *= params['e_scale']


                # Compute the loss value for this minibatch.
                loss_value = loss_fn(e_data, pred)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, network.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, network.trainable_weights))

            # Update the loss and error metrics
            train_loss_metric.update_state(e_data, pred)
            train_err_metric.update_state(e_data, pred)



            # Log every 100 batches.
            if step == 0:
                print(f"Initial loss (for one batch): {float(loss_value)}")
                print(f"Seen so far: {((step + 1) * batch_size)} molecules")

                # Reset the weights for different batch sizes
                network.save_weights('initial_weights.h5')

                # with train_summary_writer.as_default():
                #     tf.summary.scalar('Training loss', train_loss_metric.result(), step=step)
                #     tf.summary.scalar('Training error', train_err_metric.result(), step=step)


            elif step % 20 == 0:
                print(f"Training loss (for one batch) at step {step}: {float(loss_value)}")
                print(f"Seen so far: {((step + 1) * batch_size)} molecules")
                hund_step_times += [(time.time() - start_time_epoch)]
                print(f'Training time for 20 batches: {((hund_step_times[-1] - hund_step_times[-2]) if len(hund_step_times) > 1 else hund_step_times[-1])} s')

                # Record tensorboad metrics
                # with train_summary_writer.as_default():
                #     tf.summary.scalar('Training loss', train_loss_metric.result(), step=step)
                #     tf.summary.scalar('Training error', train_err_metric.result(), step=step)
                #     tf.summary.scalar('Training time/20 batches', ((hund_step_times[-1] - hund_step_times[-2]) if len(hund_step_times) > 1 else hund_step_times[-1]), step=step)


        print(f'Training time for epoch {epoch + 1}: {(time.time() - start_time_epoch)} s')



        # #Update the training metric now that the network has been trained
        # print(f'Calculating training error for epoch {(epoch + 1)}')
        # for batch in train_set:
        #     pred = network(batch, training=False)  # Logits for this minibatch

        #     ind = batch['ind_1']
        #     nbatch = tf.reduce_max(ind)+1
        #     pred = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)
        #     train_err_metric.update_state(e_data, pred)
        
        

        

        # Run a validation loop at the end of each epoch
        print(f'Starting validation for epoch {(epoch + 1)}')
        for step, batch in enumerate(test_set):
            val_pred = network(batch, training=False)
            ind = batch['ind_1']
            nbatch = tf.reduce_max(ind)+1
            val_pred = tf.math.unsorted_segment_sum(val_pred, ind[:, 0], nbatch)
            e_data = batch['e_data']

            if params['e_dress']:
                e_data -= atomic_dress(batch, params['e_dress'], dtype=pred.dtype)
                e_data *= params['e_scale']


            # Update val metrics
            val_loss_metric.update_state(e_data, val_pred)
            val_err_metric.update_state(e_data, val_pred)

            # Record Tensorboard metrics
            # if step % 5 == 0:
                
            #     with test_summary_writer.as_default():
            #         tf.summary.scalar('Validation loss', val_loss_metric.result(), step=step)
            #         tf.summary.scalar('Validation error', val_err_metric.result(), step=step)

        print(f"Time taken for epoch {epoch + 1}: {(time.time() - start_time_epoch)} s")

        # Display metrics at the end of each epoch
        train_err = train_err_metric.result()
        print(f"Training err over epoch {(epoch + 1)}: {float(train_err)}")
        val_err = val_err_metric.result()
        print(f"Validation err for epoch {(epoch + 1)}: {float(val_err)}")
        # Reset training metrics at the end of each epoch        
        train_err_metric.reset_states()
        val_err_metric.reset_states()
        
        




def _generator(molecule):
        data = {'coord': molecule.positions,
                'ind_1': np.zeros([len(molecule), 1]),
                'elems': molecule.numbers}
        yield data

def predict_energy(molecule, network=None, params=None):
        '''Takes an ASE Atoms object and outputs PiNet's energy prediction'''
        dtype=tf.float32
        dtypes = {'coord': dtype, 'elems': tf.int32, 'ind_1': tf.int32}
        shapes = {'coord': [None, 3], 'elems': [None], 'ind_1': [None, 1]}

        pred_dataset = tf.data.Dataset.from_generator(lambda:_generator(molecule), dtypes, shapes)

        for molecule in pred_dataset:
                molecule = network.preprocess(molecule)
                pred = network(molecule, training=False)
                ind = molecule['ind_1']
                nbatch = tf.reduce_max(ind)+1
                pred = pred/params['e_scale']
                if params['e_dress']:
                        pred += atomic_dress(molecule, params['e_dress'], dtype=pred.dtype)
                energy_prediction = tf.math.unsorted_segment_sum(pred, ind[:, 0], nbatch)
                energy_prediction_numpy = energy_prediction.numpy()[0]
        return energy_prediction_numpy

        

def shifted_softplus(x):
    return K.log((0.5*K.exp(x) + 0.5))