import keras
from csbdeep.internals import nets
from csbdeep.models import Config
import model_builder
import tensorflow as tf
import care

def build_wunet(config, 
                input_shape = (50,128,128,4)):
    print('=== Building WU-net Model --------------------------------------------')
    input = keras.layers.Input(input_shape)
    LL, LH, HL, HH = tf.split(input,4, axis = 3)
    # Building four CARE UNETs for each frequency band
    LLmodel = care.build_care(config, 'SXYC')(LL)
    LHmodel = care.build_care(config, 'SXYC')(LH)
    HLmodel = care.build_care(config, 'SXYC')(HL)
    HHmodel = care.build_care(config, 'SXYC')(HH)
    # Connect trained models
    concat_model = keras.layers.Concatenate()([LLmodel, LHmodel, HLmodel,HHmodel])
    model = keras.Model(input, concat_model, name='WU-net')
    print('--------------------------------------------------------------------')

    return model

def build_and_compile_WUnet(config):
    config['input_shape'] = [128,128]
    wunet = build_wunet(config,
                (*config['input_shape'], 4))

    wunet = model_builder.compile_model(wunet, config['initial_learning_rate'], config['loss'], config['metrics'])
    return wunet