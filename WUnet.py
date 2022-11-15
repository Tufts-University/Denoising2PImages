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
    caremodel = care.build_care(config, 'SXYC')


    LLmodel = caremodel(LL)
    LHmodel = caremodel(LH)
    HLmodel = caremodel(HL)
    HHmodel = caremodel(HH)
    # Connect trained models
    concat_model = keras.layers.Concatenate()([LLmodel, LHmodel, HLmodel,HHmodel])
    model = keras.Model(input, concat_model, name='WU-net')
    print('--------------------------------------------------------------------')

    return model

def build_and_compile_WUnet(config):
    wunet = build_wunet(config,
                (*config['input_shape'], 4))

    wunet = model_builder.compile_model(wunet, config['initial_learning_rate'], config['loss'], config['metrics'])
    return wunet