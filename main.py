import os as os 
import tensorflow as ts
import keras as keras

#################################################################################
''' Data was Normalized in Local Prep'''

data_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/NV_713_FAD_healthy.npz'
model_name = 'FAD_model_0713_cervix_SSIML1'
main_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/'
os.chdir(main_path)
if not os.path.exists(os.path.join(main_path,model_name)): os.mkdir(os.path.join(main_path,model_name))
model_save_path = os.path.join(main_path,model_name)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(is_cuda_gpu_available)
(X,Y), (X_val,Y_val) ,axes = load_training_data(data_path, validation_split=0.15, axes='SXY', verbose=True)
#c = axes_dict(axes)['C']
#n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

config = {
    "epochs": 300,
    "steps_per_epoch": 4,
    "num_residual_groups": 5,
    "num_residual_blocks": 5,
    "input_shape": [256, 256],
    "training_data_images": {"raw":X,
                          "gt":Y},
    "validation_data_images": {"raw":X_val,
                            "gt":Y_val},
    "LW_weight_seed": 0,
    "LW_weight_num": 4
}

config.setdefault('epochs', 300)
config.setdefault('steps_per_epoch', 256)
config.setdefault('initial_learning_rate', 1e-5)
config.setdefault('loss','SSIML1_loss')
config.setdefault('metrics', ['psnr','ssim'])
config.setdefault('num_residual_blocks', 3)
config.setdefault('num_residual_groups', 5)
config.setdefault('channel_reduction', 4)
config.setdefault('num_channels', 32)
config.setdefault('LW_weight_seed', 0)
config.setdefault('LW_weight_num', 4)

input_shape = config['input_shape']
print('Building RCAN model')
print('  - input_shape =', input_shape)
for s in ['num_channels',
          'num_residual_blocks',
          'num_residual_groups',
          'channel_reduction']:
    print(f'  - {s} =', config[s])

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = build_rcan(
        (*input_shape,1),
        num_channels=config['num_channels'],
        num_residual_blocks=config['num_residual_blocks'],
        num_residual_groups=config['num_residual_groups'],
        channel_reduction=config['channel_reduction'])
        
    gpus = get_gpu_count()
    
    #model = convert_to_multi_gpu_model(model, gpus)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=config['initial_learning_rate']),
        loss={'mae' : mae,'mse': mse,'SSIM_loss': SSIM_loss,'SSIML1_loss': SSIML1_loss}[config['loss']],
        metrics=[{'psnr': psnr, 'ssim': ssim}[m] for m in config['metrics']])

data_gen = DataGenerator(
    input_shape,
    50,
    transform_function=None)

training_data = data_gen.flow(*list(zip([X,Y])))
validation_data = data_gen.flow(*list(zip([X_val,Y_val])))

if validation_data is not None:
    checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
else:
    checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'
    
steps_per_epoch = config['steps_per_epoch'] //gpus
validation_steps = None if validation_data is None else steps_per_epoch

output_dir = model_save_path

print('Training RCAN model')
model.fit(
    x=training_data,
    epochs=config['epochs'],
    #steps_per_epoch=steps_per_epoch,
    validation_data=validation_data,
    #validation_steps=validation_steps,
    verbose=0,
    callbacks=[
        keras.callbacks.LearningRateScheduler(
            staircase_exponential_decay(config['epochs'] // 4)),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir),
            write_graph=True),
        ModelCheckpoint(
            str( pathlib.Path(output_dir) / checkpoint_filepath),
            monitor='loss' if validation_data is None else 'val_loss',
            save_best_only=True, verbose = 1, mode='min'),
        TqdmCallback()
    ])
    