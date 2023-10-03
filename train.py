import pathlib
import os
import shutil
import tensorflow as tf
import numpy as np

# Local dependencies
import callbacks
import model_builder
import data_generator
import basics
import metrics
import srgan

def determine_training_strategy(model, output_dir):
    print('=== Determining Training Strategy -----------------------------------')

    if not os.path.exists(output_dir):
        print(f'Creating output directory: "{output_dir}"')
        os.makedirs(output_dir)

    dir_contents = os.listdir(output_dir)

    checkpoint_files = [filename for filename in dir_contents if 'weights' in filename.lower()]
    finished_training = basics.final_weights_name() in dir_contents

    if finished_training:
        raise Exception(f'Model has already trained and produced final weights: "{basics.final_weights_name()}"')
    elif len(checkpoint_files) > 0:
        print(f'Found {len(checkpoint_files)} checkpoint weight files: {checkpoint_files}.')
        last_modified_file = max(checkpoint_files, key=lambda file: os.path.getmtime(os.path.join(output_dir, file)))
        print(f'Found last modified checkpoint file: "{last_modified_file}"')
        model.load_weights(os.path.join(output_dir, last_modified_file))
        print("Successfully loaded weights from last checkpoint.")
    else: 
        print('Starting training without any checkpoint weights.')

    print('--------------------------------------------------------------------')

    return model

def fit_model(model, model_name, config, output_dir, training_data, validation_data):
    print('=== Fitting model --------------------------------------------------')
    final_dir = pathlib.Path(output_dir)
    os.chdir(final_dir)
    if validation_data is not None:
        checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
    else:
        checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'
    x = training_data[0]
    y = training_data[1]
    model.fit(
        x=x,
        y=y,
        epochs=config['epochs'],
        shuffle=True,
        validation_data=validation_data,
        verbose=0,
        callbacks=callbacks.get_callbacks(
            model_name,
            config['epochs'],
            final_dir,
            checkpoint_filepath,
            validation_data))

    print('--------------------------------------------------------------------')

    return model

def final_image_generator(images,config):
    if config['wavelet_function'] != "":
        wavelet_config = data_generator.get_wavelet_config(function_name=config['wavelet_function'])
        restored_images = data_generator.wavelet_inverse_transform(images,wavelet_config)
        return tf.convert_to_tensor(restored_images)
    else: 
        return  tf.convert_to_tensor(images)
@tf.function()
def train_step(model,optimizer,loss_fn,train_metrics, data,config):
    with tf.GradientTape() as tape:
        eval_metrics = metrics.lookup_metrics(config['metrics'])
        X_N = data['NADH'][0]
        Y_N = data['NADH'][1]
        Y_N = final_image_generator(Y_N,config)
        
        X_F = data['FAD'][0]
        Y_F = data['FAD'][1]
        Y_F = final_image_generator(Y_F,config)

        if config['training_data_type'] == 'NADH':
            logits = model(X_N, training=True)
            logits = final_image_generator(logits,config)

            logits2 = model(X_F, training=False)
            logits2 = final_image_generator(logits2,config)

            loss_value = loss_fn((Y_N,Y_F), (logits,logits2))
            training_y = Y_N
        else:
            logits = model(X_F, training=True)
            logits = final_image_generator(logits,config)

            logits2 = model(X_N, training=False)
            logits2 = final_image_generator(logits2,config)

            loss_value = loss_fn((Y_N,Y_F), (logits2,logits))
            training_y = Y_F
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    metrics_eval = {}
    for i in range(len(eval_metrics)):
        train_metrics[i].update_state(eval_metrics[i](training_y, logits))
        metrics_eval[config['metrics'][i]] = train_metrics[i].result()
    return loss_value,metrics_eval,optimizer

def test_step(model,loss_fn,val_metrics,data,config):
    eval_metrics = metrics.lookup_metrics(config['metrics'])
    X_N = data['NADH'][0]
    Y_N = data['NADH'][1]
    Y_N = final_image_generator(Y_N,config)
    
    X_F = data['FAD'][0]
    Y_F = data['FAD'][1]
    Y_F = final_image_generator(Y_F,config)

    if config['training_data_type'] == 'NADH':
        logits = model(X_N, training=True)
        logits = final_image_generator(logits,config)

        logits2 = model(X_F, training=False)
        logits2 = final_image_generator(logits2,config)

        loss_value = loss_fn((Y_N,Y_F), (logits,logits2))
        val_y = Y_N
    else:
        logits = model(X_F, training=True)
        logits = final_image_generator(logits,config)

        logits2 = model(X_N, training=False)
        logits2 = final_image_generator(logits2,config)

        loss_value = loss_fn((Y_N,Y_F), (logits2,logits))
        val_y = Y_F

    metrics_val = {}
    for i in range(len(val_metrics)):
        val_metrics[i].update_state(eval_metrics[i](val_y, logits))
        metrics_val[config['metrics'][i]] = val_metrics[i].result()
    return loss_value, metrics_val

def fit_RR_model(model, model_name, config, output_dir, training_data, validation_data,strategy):
    print('=== Fitting model --------------------------------------------------')
    final_dir = pathlib.Path(output_dir)
    os.chdir(final_dir)
    if validation_data is not None:
        checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
    else:
        checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'

    # Initialize model training loops
    with strategy.scope():   
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['initial_learning_rate'])
        loss_fn = metrics.lookup_loss('RR_loss')
        eval_metrics = metrics.lookup_metrics(config['metrics'])
        _callbacks = callbacks.get_callbacks(
                model_name,
                config['epochs'],
                final_dir,
                checkpoint_filepath,
                validation_data)
        callback = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)
        logs = {}
        callback.on_train_begin(logs=logs)
        x_N, y_N, x_F, y_F  = training_data['NADH'][0], training_data['NADH'][1], training_data['FAD'][0], training_data['FAD'][1]
        x_val_N, y_val_N, x_val_F, y_val_F  = validation_data['NADH'][0], validation_data['NADH'][1], validation_data['FAD'][0], validation_data['FAD'][1]

        all_training_data = data_generator.RR_loss_Generator(x_N,y_N,x_F,y_F,config['batch_size'],config,True)
        all_val_data = data_generator.RR_loss_Generator(x_val_N,y_val_N,x_val_F,y_val_F,config['batch_size'],config,False)
        
        train_metrics = [tf.keras.metrics.Mean()]*len(eval_metrics)
        val_metrics = [tf.keras.metrics.Mean()]*len(eval_metrics)
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()


        for epoch in range(config['epochs']):
            callback.on_epoch_begin(epoch, logs=logs)
            # Training Loop
            for i, data in enumerate(all_training_data):
                callback.on_batch_begin(i, logs=logs)
                callback.on_train_batch_begin(i,logs=logs)
                loss_val, train_metrics,optimizer = strategy.run(train_step, args=(model,optimizer,loss_fn,train_metrics,data,config))
                train_loss.update_state(loss_val)
                logs["train_loss"] = train_loss.result()
                for metric_name in config['metrics']:
                    logs[metric_name] = train_metrics[metric_name]
                callback.on_train_batch_end(i, logs=logs)
                callback.on_batch_end(i, logs=logs)
            # Reset training metrics at the end of each epoch
            for i in range(len(train_metrics)):
                train_metrics[i].reset_states()

            # Validation Loop
            for i, data in enumerate(all_val_data):
                callback.on_batch_begin(i, logs=logs)
                callback.on_test_batch_begin(i, logs=logs)
                loss_val, val_metrics = test_step(model,loss_fn,val_metrics, data,config)
                val_loss.update_state(loss_val)
                logs["val_loss"] = train_loss.result()
                for metric_name in config['metrics']:
                    logs[metric_name] = val_metrics[metric_name]
                callback.on_test_batch_end(i, logs=logs)
                callback.on_batch_end(i, logs=logs)
            
            # Reset training metrics at the end of each epoch
            for i in range(len(val_metrics)):
                val_metrics[i].reset_states()

            callback.on_epoch_end(epoch, logs=logs)
        callback.on_train_end(logs=logs)
    print('--------------------------------------------------------------------')

    return model

def train(model_name, config, output_dir, data_path):
    print('Training...')

    if config['all_data']==1:
        # Need to load both NADH and FAD data and combine them for training
        if config['training_data_type'] == '':
            train = np.empty([2,1,256,256,1])
            val = np.empty([2,1,256,256,1])
            for path in data_path:
                (training_data, validation_data) = data_generator.gather_data(
                    config, 
                    path, 
                    requires_channel_dim=model_name == 'care' or model_name == 'wunet' or model_name == 'UnetRCAN')
                train = np.concatenate((train,training_data),axis=1)
                val = np.concatenate((val,validation_data),axis=1)
            training_data = train[:,1:,:,:,:]
            validation_data = val[:,1:,:,:,:]

            print(f'Got final training data with shape {np.shape(training_data)}.')
            print(f'Got final validation data with shape {np.shape(validation_data)}.')

            print('----------------------------------------------------------------------')
        else:
            # Loading NADH and FAD for RR Loss calculation
            train = []
            val = []
            for path in data_path:
                (training_data, validation_data) = data_generator.gather_data(
                    config, 
                    path, 
                    requires_channel_dim=model_name == 'care' or model_name == 'wunet' or model_name == 'UnetRCAN')
                train.append(training_data)
                val.append(validation_data)
            NADH_tr_data = train[0]
            FAD_tr_data = train[1]
            NADH_va_data = val[0]
            FAD_va_data = val[1]
            print(f'Got final training data with shape {np.shape(NADH_tr_data)}.')
            print(f'Got final validation data with shape {np.shape(NADH_va_data)}.')

            print('----------------------------------------------------------------------')
            training_data = {'NADH':NADH_tr_data, 'FAD':FAD_tr_data}
            validation_data = {'NADH':NADH_va_data, 'FAD':FAD_va_data}
    else:
        (training_data, validation_data) = data_generator.gather_data(
            config, 
            data_path, 
            requires_channel_dim=model_name == 'care' or model_name == 'wunet' or model_name == 'UnetRCAN')

    strategy = model_builder.create_strategy()
    if model_name == 'srgan':
        initial_path = os.getcwd()
        if not os.path.exists(pathlib.Path(output_dir)):
            print(f'Creating output directory: "{output_dir}"')
            os.makedirs(output_dir)
        srgan_checkpoint, srgan_checkpoint_manager = srgan.SRGAN_fit_model(model_name, strategy, config, initial_path,output_dir,training_data, validation_data)
        os.chdir(pathlib.Path(output_dir)/ 'ckpt' / 'srgan')
        srgan_checkpoint.restore(srgan_checkpoint_manager.latest_checkpoint)
        final_model = srgan_checkpoint.generator
        final_weights_path = str(pathlib.Path(output_dir) / basics.final_weights_name())
        final_model.save_weights(final_weights_path)
        print(f'Weights are saved to: "{final_weights_path}"')
    else:
        initial_path = os.getcwd()
        model = model_builder.build_and_compile_model(model_name, strategy, config)
        model = determine_training_strategy(model, output_dir)
        if config['loss'] == 'RR_loss':
            model = fit_RR_model(model, model_name, config, output_dir,
                            training_data, validation_data,strategy)
        else:
            model = fit_model(model, model_name, config, output_dir,
                            training_data, validation_data)

        os.chdir(output_dir)
        model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
        assert len(model_paths) != 0, f'No models found under {output_dir}'
        latest = max(model_paths, key=os.path.getmtime)
        final_weights_path = str(pathlib.Path(output_dir) / basics.final_weights_name())
        source = output_dir + '/' + latest
        print(f'Location of source file: "{source}"')
        print(f'Location of Final Weights file: "{final_weights_path}"')
        shutil.copy(source, final_weights_path)
        print(f'Weights are saved to: "{final_weights_path}"')