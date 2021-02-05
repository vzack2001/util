"""
    create keras resnet model
"""
#import warnings
#warnings.filterwarnings('ignore', module='tensorflow')

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

#import tensorflow_probability as tfp
import functools

# make layer helper functions
_conv2D = functools.partial(
    keras.layers.Conv2D,
    #keras.layers.SeparableConv2D,
    #strides=(1,1),
    #kernel_initializer='ones',
    #kernel_initializer=glorot_uniform(seed=0),
    bias_initializer='zeros',
    kernel_regularizer=keras.regularizers.l2(1.e-4),
    padding='same')

# https://github.com/raghakot/keras-resnet/blob/master/resnet.py
def _bn_relu(input, name='bn_relu'):
    """ Helper to build a BN -> relu block
    """
    with tf.name_scope(name):
        input = keras.layers.BatchNormalization(axis=-1)(input)
        return keras.layers.Activation('relu')(input)

def _conv_bn_relu(**conv_params):
    """ Helper to build a conv -> BN -> relu block
    """
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1,1))
    padding = conv_params.setdefault('padding', 'same')
    name = conv_params.setdefault('name', 'conv_bn_relu')

    def f(input):
        with tf.name_scope(name):
            conv = _conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding)(input)
            return _bn_relu(conv)
    return f

def _bn_relu_conv(**conv_params):
    """ Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1,1))
    padding = conv_params.setdefault('padding', 'same')
    name = conv_params.setdefault('name', 'bn_relu_conv')

    def f(input):
        with tf.name_scope(name):
            activation = _bn_relu(input)
            return _conv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding)(activation)
    return f

def _shortcut(input, residual, name=None):
    """ Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[-1] == residual_shape[-1]

    name = name or f'conv2D_{input_shape[1]}x{input_shape[-1]}_{residual_shape[1]}x{residual_shape[-1]}'

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        with tf.name_scope(name):
            shortcut = _conv2D(filters=residual_shape[-1],
                            kernel_size=(1,1),
                            strides=(stride_width, stride_height),
                            padding='valid')(input)
    return keras.layers.add([shortcut, residual])

def _residual_block(block_fn, filters, num_blocks, is_first_layer=False, name='residual_block'):
    """ Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        input_shape = K.int_shape(input)
        input_shape_str = 'x'.join(map(str, input_shape[2:]))

        with tf.name_scope(name):
            for i in range(num_blocks):
                init_strides = (1,1)
                if i == 0 and not is_first_layer:
                    init_strides = (2,2)
                input = block_fn(filters=filters,
                                init_strides=init_strides,
                                is_first_block_of_first_layer=(is_first_layer and i==0),
                                name=f'{block_fn.__name__}_{input_shape_str}')(input)

                input_shape = K.int_shape(input)
                input_shape_str = 'x'.join(map(str, input_shape[2:]))

            return input

    return f

def basic_block(filters, init_strides=(1,1), is_first_block_of_first_layer=False, name='basic_block'):
    """ Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        with tf.name_scope(name):
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv1 = _conv2D(filters=filters,
                            kernel_size=(3,3),
                            strides=init_strides)(input)
            else:
                conv1 = _bn_relu_conv(filters=filters,
                            kernel_size=(3,3),
                            strides=init_strides)(input)
            residual = _bn_relu_conv(filters=filters, kernel_size=(3,3))(conv1)
            return _shortcut(input, residual)
    return f

def bottleneck(filters, init_strides=(1,1), is_first_block_of_first_layer=False, name='bottleneck'):
    """ Bottleneck architecture for > 34 layer resnet.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        Returns:
            A final conv layer of filters * 4
    """
    def f(input):
        with tf.name_scope(name):
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv_1x1 = _conv2D(filters=filters,
                                kernel_size=(1,1),
                                strides=init_strides)(input)
            else:
                conv_1x1 = _bn_relu_conv(filters=filters,
                                kernel_size=(1,1),
                                strides=init_strides)(input)
            conv_3x3 = _bn_relu_conv(filters=filters, kernel_size=(3,3))(conv_1x1)
            residual = _bn_relu_conv(filters=filters*4, kernel_size=(1,1))(conv_3x3)
            return _shortcut(input, residual)
    return f

# slice & stats helper functions
def _value_range(values, percentile=[10,90], treshold=0, name='value_range'):
    """ input
            values (?,1)
        return
            value_range (2,?,1)
    """
    with tf.name_scope(name):
        #value_percentile = tfp.stats.percentile(values, percentile, axis=-1) # (2,?) /2 -- [min, max]/
        value_percentile = _percentile(values, percentile, axis=-1)          # (2,?) /2 -- [min, max]/
        value_last = values[:,-1]                                            # (?,)
        value_range = tf.stack([
            tf.minimum(value_percentile[0], value_last - treshold),          # (2,?) /2 -- [min, max]/
            tf.maximum(value_percentile[1], value_last + treshold)
            ], axis=0)
        return tf.expand_dims(value_range, axis=-1)  # (2,?,1)

def _digitize(values, value_range, nbins, name='digitize'):
    """ tf analog numpy digitize()
    """
    with tf.name_scope(name):
        scaled_values = tf.truediv(
            values - value_range[0],
            value_range[1] - value_range[0])
        indices = tf.floor(scaled_values * (nbins-1))
        indices = tf.clip_by_value(indices, -1, nbins-1) + 1
        return tf.cast(indices, tf.int32)

def _bincount(targets, minlength=None, axis=None, name='bincount'):
    """ analog tf.math.bincount() function
        use axis are avalable in tf(v2.4)
    """
    with tf.name_scope(name):
        if minlength is None:
            minlength = tf.reduce_max(targets) + 1
        return tf.reduce_sum(tf.one_hot(targets, minlength), axis=axis)

def _histogram(values, value_range, nbins, name='histogram'):
    """ histogram
    """
    with tf.name_scope(name):
        targets = _digitize(values, value_range, nbins)
        hist = _bincount(targets, minlength=nbins+1, axis=1)  # (?,16)
        #hist = tf.math.bincount(targets, minlength=nbins+1, axis=1)  # for use `axis` upgrade needed
        hist = hist / tf.expand_dims(tf.reduce_max(hist, axis=-1), axis=-1)
        return tf.expand_dims(hist, axis=-1)

def _percentile(x, q, axis=None, name='percentile'):
    """ from tensorflow_probability stats.percentile
        interpolation='nearest'
        axis=None (-1)
    """
    with tf.name_scope(name):
        axis = axis or -1
        k = tf.shape(x)[-1]
        d = tf.cast(k, tf.float64)
        q = tf.cast(q, tf.float64)
        sorted_x, _ = tf.math.top_k(x, k=k)
        frac_at_q_or_above = 1. - q / 100.
        indices = tf.round((d - 1) * frac_at_q_or_above)  # 'nearest'
        indices = tf.clip_by_value(tf.cast(indices, tf.int32), 0, k - 1)
        gathered_x = tf.gather(sorted_x, indices, axis=axis)  # (?,len(q))
        return tf.transpose(gathered_x, [1,0])  # (len(q),?)

def _handle_global_vars(bins_csv='bins_16.csv'):
    global lv_name
    global lv_cols
    global lv_time
    global db_bins
    #                0       1        2        3        4         5       6        7
    lv_name = [  'MN1',   'W1',    'D1',    'H4',    'H1',    'M15',   'M5',    'M1' ]
    lv_cols = ['LVMN1', 'LVW1',  'LVD1',  'LVH4',  'LVH1',  'LVM15', 'LVM5',  'LVM1' ]
    lv_time = [  28800,   7200,    1440,     240,      60,       15,      5,       1 ]

    db_bins = pd.read_csv(bins_csv, header=0, index_col=0, dtype='float32')
    pass  # _handle_global_vars()

def create_prep(inputs, depth=5, treshold=0.050, p_range=[5, 95]):
    """ prepare data inputs
            from shape (?,28800,10)
            to shape (?,16,12,5)

        use tensorflow.python.ops math_ops._bucketize,
            #tensorflow_probability stats.percentile,
            _percentile
            _value_range,
            _digitize,
            _bincount,
            _histogram

        _handle_global_vars(bins_csv='data/bins_16.csv')
            bins_csv = 'data/bins_16.csv'
            db_bins = pd.read_csv(bins_csv, header=0, index_col=0, dtype='float32')
            #print(db_bins)
            #                0       1        2        3        4         5       6        7
            lv_name = [  'MN1',   'W1',    'D1',    'H4',    'H1',    'M15',   'M5',    'M1' ]
            lv_cols = ['LVMN1', 'LVW1',  'LVD1',  'LVH4',  'LVH1',  'LVM15', 'LVM5',  'LVM1' ]
            lv_time = [  28800,   7200,    1440,     240,      60,       15,      5,       1 ]
    """

    nbins = db_bins.shape[0]  # nb_classes-1  # 15
    levels = []
    ranges = []

    for i in range(depth):
        time = [lv_time[i], lv_time[i+1]]
        name = [lv_name[i], lv_name[i]+'_'+lv_name[i+1]]

        with tf.name_scope(name[0]):
            data = inputs[:,-time[0]:,:]
            #idx_test = tf.stack([data[:,0,-2], data[:,-1,-2]], axis=1)

            bins = db_bins[name[0]].values.tolist()
            #print_ndarray('bins = db_bins[{}].values'.format(name[0]), bins,  count=16, frm='%8.3f')  # (15,) (nbin,)
            lv_slow = tf.stack([
                data[:,0,i],
                data[:,-1,i],
                ], axis=1)                      # (?,2)
            #tft.apply_buckets
            #https://www.tensorflow.org/tfx/transform/api_docs/python/tft/apply_buckets
            lv_slow = math_ops._bucketize(lv_slow, boundaries=bins, name='bucketize_slow')

            bins = db_bins[name[1]].values.tolist()
            #print_ndarray('bins = db_bins[{}].values'.format(name[1]), bins,  count=16, frm='%8.3f')  # (15,) (nbin,)
            lv_fast = tf.stack([
                data[:,-time[1],i],
                data[:,-1,i],
                data[:,-time[1],i+1],
                data[:,-1,i+1],
                ], axis=1)                        # (?,4)
            lv_fast = math_ops._bucketize(lv_fast, boundaries=bins, name='bucketize_fast')

            price_values = data[:,:,-1]
            value_range = _value_range(price_values, percentile=p_range, treshold=treshold)  # (2,?,1)  (min|max,?,1)
            #x = tf.transpose(value_range, [1,2,0])  # just as output test case

            with tf.name_scope('stack_range'):
                last_value = price_values[:,-1]   # (?,)  last
                price_range = tf.stack([
                    last_value - treshold,
                    last_value,
                    last_value + treshold,
                    tf.reduce_min(price_values, axis=-1, name='min_slow'),
                    tf.reduce_max(price_values, axis=-1, name='max_slow'),
                    tf.reduce_min(price_values[:,-time[1]:], axis=-1, name='min_fast'),
                    tf.reduce_max(price_values[:,-time[1]:], axis=-1, name='max_fast'),
                    ], axis=1)                    # (?,7)
            price_range = _digitize(price_range, value_range, nbins, name='digitize_range')

            x = tf.concat([
                lv_slow,                          # (?,2)  slow (slow (first, last))
                lv_fast,                          # (?,4)  fast (slow (first, last) fast (first, last))
                price_range,                      # (?,7)
                ], axis=-1)                       # (?,13)

            x = tf.concat([
                _histogram(price_values, value_range, nbins, name='histogram_slow'),
                _histogram(price_values[:,-time[1]*2:-time[1]], value_range, nbins, name='histogram_fast1'),
                _histogram(price_values[:,-time[1]:], value_range, nbins, name='histogram_fast0'),
                tf.one_hot(x, nbins+1, axis=1),   # (?,16,13)
                ], axis=-1)                       # (?,16,3+13)

            levels.append(x)
            ranges.append(value_range)

    x = tf.stack(levels, axis=-1)                  #(?,16,12,5)  # (?,nclasses,channels,depth)

    return x, tf.transpose(tf.concat(ranges, axis=-1), [1,0,2])  #(?,2,5)      # (?,2,depth) min|max


class ResnetBuilder(object):

    @staticmethod
    def build(input_shape, bins_csv='bins_16.csv', depth=5, name='resnet'):
        """
        """
        _handle_global_vars(bins_csv=bins_csv)

        print(f'\n-- ResnetBuilder.build(input_shape={input_shape}, bins_csv={bins_csv}, depth={depth}, name={name})\n')

        inputs = keras.layers.Input(input_shape, name='input')
        x = inputs

        x, value_ranges = keras.layers.Lambda(create_prep, trainable=False, arguments={'depth':depth}, name='preprocessing')(x)
        #x = keras.layers.Activation(activation='linear', name='output')(x)

        a = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x)  # output_shape=(input_shape-pool_size+1)/strides)
        b = keras.layers.DepthwiseConv2D(kernel_size=1, depth_multiplier=8, padding='same')(x)
        x = keras.layers.concatenate([a,b], axis=-1)

        filters = 64
        for i, n in enumerate([3, 3, 3]):
            x = _residual_block(basic_block, filters=filters, num_blocks=n, is_first_layer=(i==0))(x)
            filters *= 2

        with tf.name_scope('post_pr'):
            # Last activation
            x = _bn_relu(x)

            # Classifier block
            block_shape = K.int_shape(x)
            x = keras.layers.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1))(x)
            #x = keras.layers.Flatten()(x)
            x = keras.layers.Reshape((2,128))(x)
            x = keras.layers.Dense(units=11, activation="softmax")(x)

        x = keras.layers.Activation(activation='linear', name='output')(x)

        model = keras.Model(inputs, x, name=name)

        model.summary()
        print('model.inputs :', model.inputs)
        print('model.outputs:', model.outputs)
        print()

        return model

    pass  # class ResnetBuilder


if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()
    print('tensorflow version: {0}'.format(tf.__version__))
    print('keras version: {0}'.format(keras.__version__))
    print('tf.executing_eagerly(): {}'.format(tf.executing_eagerly()))

    # define model name and path
    model_name = 'resnet'

    data_shape = (28800,10)  # np.shape(x)[1:]
    print('data_shape:', data_shape)

    # create model
    model = ResnetBuilder.build(data_shape, bins_csv='bins_16.csv', name=model_name)

    # https://medium.com/analytics-vidhya/basics-of-using-tensorboard-in-tensorflow-1-2-b715b068ac5a
    logdir = 'logs'
    tf.compat.v1.summary.FileWriter(logdir, graph=tf.compat.v1.get_default_graph()).close()

    pass