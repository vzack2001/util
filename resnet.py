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
        return keras.layers.Activation('relu', alpha=0.0)(input)

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
    """
    #                0       1        2        3        4         5       6        7
    lv_name = [  'MN1',   'W1',    'D1',    'H4',    'H1',    'M15',   'M5',    'M1' ]
    lv_cols = ['LVMN1', 'LVW1',  'LVD1',  'LVH4',  'LVH1',  'LVM15', 'LVM5',  'LVM1' ]
    lv_time = [  28800,   7200,    1440,     240,      60,       15,      5,       1 ]

    _cols = ['percentile','MN1','W1','D1','H4','H1','M15','M5','M1','MN1_W1','W1_D1','D1_H4','H4_H1','H1_M15','M15_M5','M5_M1']
    _rows = [[1.0,-2.447450723648071,-4.3177623939514165,-6.3557000160217285,-8.466059684753418,-9.868930702209472,-11.187950134277344,-12.33390998840332,-15.201800346374512,-4.037990093231201,-6.00206995010376,-8.103440284729004,-9.51678147315979,-10.829000473022461,-11.951820373535156,-14.190199851989746],
        [2.5,-2.2133100032806396,-3.9119200706481934,-5.871469974517822,-7.971982097625732,-9.357959747314453,-10.654029846191406,-11.706509590148926,-13.815509796142578,-3.5091800689697266,-5.452270030975342,-7.540170192718506,-8.959269523620605,-10.258740425109863,-11.315460205078125,-13.004579544067383],
        [5.0,-1.9093300104141235,-3.4993300437927246,-5.441074085235595,-7.532569885253906,-8.907859802246094,-10.191619873046875,-11.18002986907959,-12.89922046661377,-3.0329699516296387,-4.952859878540039,-7.037650108337402,-8.468887567520142,-9.760040283203125,-10.788009643554688,-12.169260025024414],
        [10.0,-1.5410100221633911,-3.020400047302246,-4.938948059082031,-7.022890090942383,-8.384160041809082,-9.650349617004395,-10.598739624023438,-11.982930183410645,-2.5604898929595947,-4.4264397621154785,-6.470665216445923,-7.90254020690918,-9.177720069885254,-10.207030296325684,-11.321479797363281],
        [20.0,-1.0277600288391113,-2.524139881134033,-4.389679908752441,-6.4340901374816895,-7.755660057067871,-8.987199783325195,-9.90569019317627,-11.000849723815918,-1.9874299764633179,-3.7780098915100098,-5.784180164337158,-7.240270137786865,-8.466130256652832,-9.496930122375488,-10.596630096435547],
        [30.0,-0.693880021572113,-2.1557700634002686,-4.005080223083496,-6.031439781188965,-7.330619812011719,-8.507829666137695,-9.399689674377441,-10.596630096435547,-1.5796500444412231,-3.282870054244995,-5.247350215911865,-6.7797698974609375,-7.970870018005371,-8.972590446472168,-9.943479537963867],
        [40.0,-0.43154001235961914,-1.8716800212860107,-3.6807799339294434,-5.686970233917236,-6.986070156097412,-8.11320972442627,-8.957639694213867,-9.99368953704834,-1.2257399559020996,-2.841749906539917,-4.76255989074707,-6.3881402015686035,-7.5739898681640625,-8.527799606323242,-9.49802017211914],
        [50.0,-0.16436000168323517,-1.5963300466537476,-3.3727200031280518,-5.358520030975342,-6.670050144195557,-7.769979953765869,-8.547510147094727,-9.585029602050781,-0.873420000076294,-2.4463000297546387,-4.315420150756836,-6.023379802703857,-7.221710205078125,-8.135379791259766,-8.987199783325195],
        [60.0,0.10655000060796738,-1.3389500379562378,-3.072000026702881,-5.026080131530762,-6.355410099029541,-7.44789981842041,-8.160039901733398,-9.011489868164062,-0.5510600209236145,-2.056689977645874,-3.879390001296997,-5.650790214538574,-6.881919860839844,-7.777639865875244,-8.517189979553223],
        [70.0,0.39065998792648315,-1.0625499486923218,-2.743410110473633,-4.663980007171631,-6.015250205993652,-7.123149871826172,-7.787829875946045,-8.427579879760742,-0.20032000541687012,-1.661679983139038,-3.438509941101074,-5.243169784545898,-6.515160083770752,-7.427140235900879,-8.053689956665039],
        [80.0,0.7234299778938293,-0.7221300005912781,-2.3731698989868164,-4.230669975280762,-5.598939895629883,-6.750539779663086,-7.397550106048584,-7.957580089569092,0.2176699936389923,-1.241420030593872,-2.9493799209594727,-4.761129856109619,-6.076779842376709,-7.0424299240112305,-7.639100074768066],
        [90.0,1.1333480119705182,-0.27748000621795654,-1.8487399816513062,-3.6258018970489516,-4.991030216217041,-6.211830139160156,-6.9065118789672875,-7.37775993347168,0.7718700170516968,-0.66566002368927,-2.3125998973846436,-4.083680152893066,-5.434470176696777,-6.508309841156006,-7.100130081176758],
        [95.0,1.4636600017547607,0.13891999423503876,-1.4087659597396907,-3.1120100021362305,-4.469629764556885,-5.725886154174811,-6.487239837646484,-6.939499855041504,1.1656099557876587,-0.24150000512599945,-1.8073300123214722,-3.5262200832366943,-4.877600193023682,-6.028369903564453,-6.675849914550781],
        [97.5,1.7651900053024292,0.4791100025177002,-0.9900799989700317,-2.659019947052002,-4.001947784423827,-5.268199920654297,-6.086209774017334,-6.557980060577393,1.4778000116348267,0.16561000049114227,-1.3735699653625488,-3.034640073776245,-4.382919788360596,-5.572669982910156,-6.282489776611328],
        [99.0,2.0152900218963623,0.974839985370636,-0.4590100049972534,-2.106260061264038,-3.4134191608428948,-4.701019001007079,-5.560719966888428,-6.072020053863525,1.8153400421142578,0.6136299967765808,-0.809568512439724,-2.4642285585403405,-3.7802600860595703,-4.997039794921875,-5.770319938659668],]

    db_bins = pd.DataFrame(data=_rows, columns=_cols, dtype=np.float32)

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
    def build(input_shape, depth=5, name='resnet'):
        """
        """
        print(f'\n-- ResnetBuilder.build(input_shape={input_shape}, depth={depth}, name={name})\n')

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

        with tf.name_scope('embedding'):
            # Last activation
            x = _bn_relu(x)
            block_shape = K.int_shape(x)
            x = keras.layers.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1))(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_normalize')(x) # L2 normalize embeddings

        '''
        with tf.name_scope('post_pr'):
            # Last activation
            x = _bn_relu(x)

            # Classifier block
            block_shape = K.int_shape(x)
            x = keras.layers.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1))(x)
            #x = keras.layers.Flatten()(x)
            x = keras.layers.Reshape((2,128))(x)
            x = keras.layers.Dense(units=11, activation="softmax")(x)
        '''

        x = keras.layers.Activation(activation='linear', name='output')(x)

        model = keras.Model(inputs, x, name=name)

        model.summary()
        print('model.inputs :', model.inputs)
        print('model.outputs:', model.outputs)
        print()

        return model

    pass  # class ResnetBuilder


class ResnetEmbed(object):

    @staticmethod
    def build(input_shape, depth=5, name='resnet'):
        """
        """
        print(f'\n-- ResnetEmbed.build(input_shape={input_shape}, depth={depth}, name={name})\n')

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

        with tf.name_scope('embedding'):
            # Last activation
            x = _bn_relu(x)
            block_shape = K.int_shape(x)
            #x = keras.layers.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1))(x)
            x = keras.layers.Flatten()(x)
            #x = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_normalize')(x) # L2 normalize embeddings

        '''
        with tf.name_scope('post_pr'):
            # Last activation
            x = _bn_relu(x)

            # Classifier block
            block_shape = K.int_shape(x)
            x = keras.layers.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1))(x)
            #x = keras.layers.Flatten()(x)
            x = keras.layers.Reshape((2,128))(x)
            x = keras.layers.Dense(units=11, activation="softmax")(x)
        '''

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
    model_name = 'resnet_emb'

    data_shape = (28800,12)  # np.shape(x)[1:]
    print('data_shape:', data_shape)

    # create model
    #model = ResnetBuilder.build(data_shape, name=model_name)
    model = ResnetEmbed.build(data_shape, name=model_name)

    json_config = model.to_json()
    #print(json_config)
    with open(f'temp/{model_name}.json', mode='w') as f:
        f.write(json_config)

    # https://medium.com/analytics-vidhya/basics-of-using-tensorboard-in-tensorflow-1-2-b715b068ac5a
    logdir = 'C:\logs'
    tf.compat.v1.summary.FileWriter(logdir, graph=tf.compat.v1.get_default_graph()).close()

    pass