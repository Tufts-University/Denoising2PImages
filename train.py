from cgi import test
from tabnanny import check
import tensorflow as tf
import pathlib
import os
import shutil
import tqdm
import numpy as np

# Local dependencies
import callbacks
import model_builder
import data_generator
import basics
import srgan
import metrics

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

        raise Exception(f'Cannot continue training from checkpoints. Terminating...')
        # TODO: Implement continued training from checkpoints. (Load correct lr, epochs, and anything else that changes.)
        # model.load_weights(os.path.join(output_dir, last_modified_file))
        # print("Successfully loaded weights from last checkpoint.")
    else:
        print('Starting training without any checkpoint weights.')

    print('--------------------------------------------------------------------')

    return model

def fit_model(model, model_name, config, output_dir, training_data, validation_data):
    print('=== Fitting model --------------------------------------------------')
    steps_per_epoch = config['steps_per_epoch'] if config['steps_per_epoch'] != None else None
    validation_steps = None if validation_data is None else steps_per_epoch
    if validation_data is not None:
        checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
    else:
        checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'
    print(np.shape(training_data))
    print(np.shape(validation_data))
    model.fit(
        x= training_data if model_name != 'care' else training_data[0],
        y=None if model_name != 'care' else training_data[1],
        epochs=config['epochs'],
        # steps_per_epoch=steps_per_epoch,
        shuffle=True,
        validation_data=validation_data,
        # validation_steps=validation_steps,
        verbose=0,
        callbacks=callbacks.get_callbacks(
            model_name,
            config['epochs'],
            output_dir,
            checkpoint_filepath,
            validation_data))

    print('--------------------------------------------------------------------')

    return model


def train(model_name, config, output_dir, data_path):
    print('Training...')

    (training_data, validation_data) = data_generator.gather_data(
        config, 
        data_path, 
        requires_channel_dim=model_name == 'care')

    strategy = model_builder.create_strategy()
    #TODO (nvora01): Clean up the SRGAN implementation
    if model_name == 'srgan':
        generator, discriminator, care = model_builder.build_and_compile_model(model_name, strategy, config)
        
        initial_path = os.getcwd()
        os.chdir(output_dir)
        Gen_flag, CARE_flag = basics.SRGAN_Weight_search(initial_path, output_dir)
        if Gen_flag == 1:
            Gen_final_weights_path = str(initial_path /pathlib.Path(output_dir) / 'Pretrained.hdf5')
        else: 
            generator, Gen_final_weights_path = srgan.generator_train(generator, model_name, config, output_dir, training_data, validation_data, initial_path)
        generator.load_weights(Gen_final_weights_path)
        if CARE_flag == 1:
            CARE_final_weights_path = str(initial_path /pathlib.Path(output_dir) / 'CARE_Pretrained.hdf5')
        else: 
            raise Exception('CARE Model needs to be pretrained, please confirm you have weights for standard CARE model')
        care.load_weights(CARE_final_weights_path)

        srgan_checkpoint_dir = output_dir + '/ckpt/srgan'
        os.makedirs(srgan_checkpoint_dir, exist_ok=True)
        with strategy.scope():            
            learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
            generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

            srgan_checkpoint = tf.train.Checkpoint(psnr=tf.Variable(0.0),
                                                ssim=tf.Variable(0.0), 
                                                generator_optimizer=generator_optimizer,
                                                discriminator_optimizer=discriminator_optimizer,
                                                generator=generator,
                                                discriminator=discriminator)

            srgan_checkpoint_manager = tf.train.CheckpointManager(checkpoint=srgan_checkpoint,
                                    directory=srgan_checkpoint_dir,
                                    max_to_keep=3)
        
            if srgan_checkpoint_manager.latest_checkpoint:
                srgan_checkpoint.restore(srgan_checkpoint_manager.latest_checkpoint)
            perceptual_loss_metric = tf.keras.metrics.Mean()
            discriminator_loss_metric = tf.keras.metrics.Mean()
            psnr_metric = tf.keras.metrics.Mean()
            ssim_metric = tf.keras.metrics.Mean()
            for i in range(config['epochs']):
                for _, batch in enumerate(training_data):
                    print(np.shape(batch))
                    perceptual_loss, discriminator_loss = strategy.run(srgan.train_step, args=(batch,srgan_checkpoint,care))
                    perceptual_loss_metric(perceptual_loss)
                    discriminator_loss_metric(discriminator_loss)
                    lr = batch[0]
                    hr = batch[1]
                    sr = srgan_checkpoint.generator.predict(lr)
                    psnr_value = metrics.psnr(hr, sr)
                    ssim_value = metrics.ssim(hr, sr)
                    psnr_metric(psnr_value)
                    ssim_metric(ssim_value)
                CARE_loss = perceptual_loss_metric.result()
                dis_loss = discriminator_loss_metric.result()
                psnr_train = psnr_metric.result()
                ssim_train = ssim_metric.result()
                print(f'Training --> Epoch # {i}: CARE_loss = {CARE_loss:.4f}, Discrim_loss = {dis_loss:.4f}, PSNR = {psnr_train:.4f}, SSIM = {ssim_train:.4f}')
                perceptual_loss_metric.reset_states()
                discriminator_loss_metric.reset_states()
                psnr_metric.reset_states()
                ssim_metric.reset_states()

                srgan_checkpoint.psnr.assign(psnr_train)
                srgan_checkpoint.ssim.assign(ssim_train)
                srgan_checkpoint_manager.save()

                for _, val_batch in enumerate(validation_data):
                    lr = val_batch[0]
                    hr = val_batch[1]
                    sr = srgan_checkpoint.generator.predict(lr)
                    hr_output = srgan_checkpoint.discriminator.predict(hr)
                    sr_output = srgan_checkpoint.discriminator.predict(sr)

                    con_loss = metrics.calculate_content_loss(hr, sr, care)
                    gen_loss = metrics.calculate_generator_loss(sr_output)
                    perc_loss = con_loss + 0.001 * gen_loss
                    disc_loss = metrics.calculate_discriminator_loss(hr_output, sr_output)

                    perceptual_loss_metric(perc_loss)
                    discriminator_loss_metric(disc_loss)

                    psnr_value = metrics.psnr(hr, sr)
                    ssim_value = metrics.ssim(hr, sr)
                    psnr_metric(psnr_value)
                    ssim_metric(ssim_value)
                CARE_loss = perceptual_loss_metric.result()
                dis_loss = discriminator_loss_metric.result()
                psnr_train = psnr_metric.result()
                ssim_train = ssim_metric.result()
                print(f'Validation --> Epoch # {i}: CARE_loss = {CARE_loss:.4f}, Discrim_loss = {dis_loss:.4f}, PSNR = {psnr_train:.4f}, SSIM = {ssim_train:.4f}')
                perceptual_loss_metric.reset_states()
                discriminator_loss_metric.reset_states()
                psnr_metric.reset_states()
                ssim_metric.reset_states()
        os.chdir(output_dir)
        model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
        assert len(model_paths) != 0, f'No models found under {output_dir}'
        latest = max(model_paths, key=os.path.getmtime)
        final_weights_path = str(initial_path /pathlib.Path(output_dir) / basics.final_weights_name())
        source = str(initial_path / pathlib.Path(output_dir) / latest)
        print(f'Location of source file: "{source}"')
        print(f'Location of Final Weights file: "{final_weights_path}"')
        shutil.copy(source, final_weights_path)
        print(f'Weights are saved to: "{final_weights_path}"')
    else:
        model = model_builder.build_and_compile_model(model_name, strategy, config)
        model = determine_training_strategy(model, output_dir)
        model = fit_model(model, model_name, config, output_dir,
                        training_data, validation_data)

        initial_path = os.getcwd()
        os.chdir(output_dir)
        model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
        assert len(model_paths) != 0, f'No models found under {output_dir}'
        latest = max(model_paths, key=os.path.getmtime)
        final_weights_path = str(initial_path /pathlib.Path(output_dir) / basics.final_weights_name())
        source = str(initial_path / pathlib.Path(output_dir) / latest)
        print(f'Location of source file: "{source}"')
        print(f'Location of Final Weights file: "{final_weights_path}"')
        shutil.copy(source, final_weights_path)
        print(f'Weights are saved to: "{final_weights_path}"')