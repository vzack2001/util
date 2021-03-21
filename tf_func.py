"""slice & stats helper functions"""

import tensorflow as tf

def _value_range(values, percentile=[10,90], treshold=0, name='value_range'):
    """ input
            values (?,seq)
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


if __name__ == "__main__":

    import numpy as np
    from mylib import print_ndarray

    print('\ntensorflow helper functions')
    print('tensorflow version: {0}'.format(tf.__version__))
    print('tf.executing_eagerly(): {0}'.format(tf.executing_eagerly()))

    # define test random array
    mu = np.array([0, 2, 5, -5], dtype=np.float32).reshape(-1,1)
    sigma = np.array([1, 2, 3, 3], dtype=np.float32).reshape(-1,1)
    print_ndarray('mu', mu)
    print_ndarray('sigma', sigma)

    a_shape = (4,10000)
    a = np.random.normal(size=a_shape) * sigma + mu
    print_ndarray('a = np.random.normal(size={}) * sigma + mu'.format(a.shape), a)

    q = [25, 50, 75]
    percentile = np.percentile(a, q, axis=-1)
    print_ndarray('np.percentile(a, {}, axis=-1)'.format(q), percentile)

    percentile = _percentile(a, q, axis=-1)
    percentile = percentile.numpy()
    print_ndarray('_percentile(a, {}, axis=-1)'.format(q), percentile)

    percentile = [10,90]
    treshold = 0.0
    values = a
    print_ndarray('values = np.random.normal(size={}) * sigma + mu'.format(values.shape), values)
    value_range = _value_range(a, percentile=percentile, treshold=treshold)
    value_range = value_range.numpy()
    print_ndarray('value_range = _value_range(values, percentile={}, treshold={})'.format(percentile, treshold), value_range, count=0)
    print_ndarray('', value_range[...,0])

    nbins = 10
    h = _histogram(values, value_range, nbins=nbins)
    print_ndarray('h = _histogram(values, value_range, nbins={})'.format(nbins), h.numpy()[...,0])

    pass # main