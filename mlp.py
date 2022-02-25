import numpy as np

import tensorflow as tf
from tensorflow import keras

from typing import Any, Tuple, List
import functools

import ml_collections


# https://towardsdatascience.com/mlp-mixer-is-all-you-need-20dbc7587fe4
def get_mixer_l16_config():
    # https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
    """ Returns Mixer-L/16 configuration. """
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-L_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_dim = 1024
    config.num_blocks = 24
    config.tokens_mlp_dim = 512
    config.channels_mlp_dim = 4096
    config.num_outputs = 3
    return config

def get_mixer_s1_config():
    """ Returns Mixer-S/16 configuration. """
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-S_1'
    config.patches = ml_collections.ConfigDict({'size': (1, 1)})
    config.hidden_dim = 128
    config.num_blocks = 8
    config.tokens_mlp_dim = 64
    config.channels_mlp_dim = 128
    config.num_outputs = 3
    return config

def get_mixer_s2_config():
    """ Returns Mixer-S/16 configuration. 256 max bathc"""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-S_2'
    config.patches = ml_collections.ConfigDict({'size': (2, 2)})
    config.hidden_dim = 256
    config.num_blocks = 16
    config.tokens_mlp_dim = 64
    config.channels_mlp_dim = 512
    config.num_outputs = 3
    return config

def get_mixer_s4_config():
    """ Returns Mixer-S/16 configuration. """
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-S_4'
    config.patches = ml_collections.ConfigDict({'size': (4, 4)})
    config.hidden_dim = 128
    config.num_blocks = 2 #4
    config.tokens_mlp_dim = 64
    config.channels_mlp_dim = 128
    config.num_outputs = 3
    return config


_dense = functools.partial(
        keras.layers.Dense,
        #kernel_regularizer=keras.regularizers.l2(1.e-4),
        #bias_regularizer=keras.regularizers.l2(1.e-4),
        #activity_regularizer=keras.regularizers.L2(l2=0.01),
    )

def MlpBlock(mlp_dim: int, name='mlp_block'):
    """ Mlp Block layer. """
    def f(inputs):
        with tf.name_scope(name):
            y = _dense(units=mlp_dim, activation='gelu')(inputs)
            y =_dense(units=inputs.shape[-1])(y)
            #y = keras.layers.Dropout(0.4)(y)
            return y
    return f

def MixerBlock(tokens_mlp_dim: int, channels_mlp_dim: int, name='mixer_block'):
    """ Mixer block layer. """
    def f(inputs):
        with tf.name_scope(name):
            x = keras.layers.Activation(activation='linear')(inputs)
            y = keras.layers.LayerNormalization()(x)
            y = keras.layers.Permute(dims=[2, 1])(y)
            y = MlpBlock(tokens_mlp_dim, name='token_mixing')(y)
            y = keras.layers.Permute(dims=[2, 1])(y)

            x = keras.layers.Add()([x, y])
            y = keras.layers.LayerNormalization()(x)
            y = MlpBlock(channels_mlp_dim, name='channel_mixing')(y)

            x = keras.layers.Add()([x, y])
            return x
    return f

def MlpMixer(
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        num_blocks: int,
        name='mlp_mixer'
    ):
    """ Mixer architecture. """
    def f(inputs):
        with tf.name_scope(name):
            x = inputs
            for _ in range(num_blocks):
                x = MixerBlock(tokens_mlp_dim, channels_mlp_dim)(x)
            return x
    return f


class AddPosEmbs(tf.keras.layers.Layer):
    """ https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/position_embedding.py
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py
    """
    def __init__(self,
            initializer="glorot_uniform",
            **kwargs):
        super().__init__(**kwargs)
        self._initializer = tf.keras.initializers.get(initializer)
        pass  # __init__()

    def get_config(self):
        config = {
            "initializer": tf.keras.initializers.serialize(self._initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # input_shape is (batch_size, seq_len, emb_dim).
        #pos_emb_shape = (1, input_shape[1], input_shape[2])
        self._position_embeddings = self.add_weight(
            "embeddings",
            shape=input_shape[1:],
            #shape=pos_emb_shape,
            initializer=self._initializer)

        super().build(input_shape)
        pass  # build()

    def call(self, inputs):
        return inputs + self._position_embeddings

    pass  # AddPosEmbs


class MlpMix(object):

    @staticmethod
    def build(
            input_shape: Any,
            config: ml_collections.ConfigDict,
            log_prob_activity_regularizer=None,  # keras.regularizers.L2(l2=0.01)
            name='mlp_mixer',
        ):
        """
        """
        print(f'\n-- MlpMix.build(input_shape={input_shape}, name={name}, {config})\n')

        #input_shape = (?, 16, 16, 36)  # (batch, height, width, channels)
        inputs = keras.layers.Input(input_shape, name='input')
        #x = inputs

        x = keras.layers.Conv2D(filters=config.hidden_dim, kernel_size=config.patches.size, strides=config.patches.size, name='stem')(inputs)

        #x = einops.rearrange(x, 'n h w c -> n (h w) c')
        #n, h, w, c = tf.shape(x)
        x = keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

        x = MlpMixer(config.tokens_mlp_dim, config.channels_mlp_dim, config.num_blocks, name='mlp_mixer')(x)

        # Classifier head
        x = keras.layers.LayerNormalization()(x)  # epsilon=1e-6
        x = keras.layers.GlobalAveragePooling1D()(x)

        feature = keras.layers.Activation(activation='linear', name='feature')(x)

        log_prob = _dense(
            units=config.num_outputs,
            activation=tf.math.log_softmax,
            activity_regularizer=log_prob_activity_regularizer,
            name='log_prob')(x)
        value = _dense(units=1, name='value')(x)

        model = keras.Model(inputs, [log_prob, value, feature], name=name)

        #model.summary()
        #print('model.inputs :', model.inputs)
        #print('model.outputs:', model.outputs)
        #print()

        return model

    pass  # MlpMix


class MlpMix2(object):
    """ MlpMix (with positional embedding)
    """

    @staticmethod
    def build(
            input_shape: Any,
            config: ml_collections.ConfigDict,
            log_prob_activity_regularizer=None,  # keras.regularizers.L2(l2=0.01)
            name='mlp_mixer',
        ):
        """
        """
        print(f'\n-- MlpMix.build(input_shape={input_shape}, name={name}, {config})\n')

        #input_shape = (?, 16, 16, 36)  # (batch, height, width, channels)
        inputs = keras.layers.Input(input_shape, name='input')
        #x = inputs

        # add positional embedding
        y = AddPosEmbs()(inputs)

        x = keras.layers.Conv2D(filters=config.hidden_dim, kernel_size=config.patches.size, strides=config.patches.size, name='stem')(y)

        #x = einops.rearrange(x, 'n h w c -> n (h w) c')
        #n, h, w, c = tf.shape(x)
        x = keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

        x = MlpMixer(config.tokens_mlp_dim, config.channels_mlp_dim, config.num_blocks, name='mlp_mixer')(x)

        # Classifier head
        x = keras.layers.LayerNormalization()(x)  # epsilon=1e-6
        x = keras.layers.GlobalAveragePooling1D()(x)

        feature = keras.layers.Activation(activation='linear', name='feature')(x)

        log_prob = _dense(
            units=config.num_outputs,
            activation=tf.math.log_softmax,
            activity_regularizer=log_prob_activity_regularizer,
            name='log_prob')(x)
        value = _dense(units=1, name='value')(x)

        model = keras.Model(inputs, [log_prob, value, feature], name=name)

        #model.summary()
        #print('model.inputs :', model.inputs)
        #print('model.outputs:', model.outputs)
        #print()

        return model

    pass  # MlpMix2 (with positional embedding)


if __name__ == "__main__":
    from mylib import print_ndarray

    #tf.compat.v1.disable_eager_execution()
    print('\ntensorflow version: {0}'.format(tf.__version__))
    print('keras version: {0}'.format(keras.__version__))
    print('tf.executing_eagerly(): {}'.format(tf.executing_eagerly()))
    #print('GPUs:', tf.config.list_physical_devices('GPU'))
    print('\nphysical_devices = tf.config.list_physical_devices("GPU"):')
    physical_devices = tf.config.list_physical_devices('GPU')
    print('\n----')
    print('list_physical_devices:', physical_devices)
    for device in physical_devices:
        details = tf.config.experimental.get_device_details(device)
        print(device, details)
        print(details['device_name'])
    print('---\n')

    input_shape = (16,16,36)
    config = get_mixer_s4_config()

    if not tf.executing_eagerly():
        logdir = 'C:\logs'
        m = MlpMix2.build(input_shape, config, name='test_model')
        tf.compat.v1.summary.FileWriter(logdir, graph=tf.compat.v1.get_default_graph()).close()
        assert False, 'write graph'

    m = MlpMix2.build(input_shape, config)

    size = (1024,16,16,36)
    x = np.random.normal(size=size)
    print_ndarray(f'x = np.random.normal(size={size})', x)

    log_prob, value, feature = m(x)
    print_ndarray('log_prob', log_prob)
    print_ndarray('value', value)
    print_ndarray('feature, mean, var', np.concatenate([feature, np.mean(feature, axis=-1, keepdims=True), np.var(feature, axis=-1, keepdims=True)], axis=-1))

