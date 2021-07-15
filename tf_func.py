"""slice & stats helper functions"""

import tensorflow as tf

epsilon = 1e-16

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
        values = tf.cast(values, dtype=tf.float32)
        value_range = tf.cast(value_range, dtype=tf.float32)
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
        max_value = targets.shape[1]
        hist = hist / max_value
        #hist = tf.math.bincount(targets, minlength=nbins+1, axis=1)  # for use `axis` upgrade needed
        #hist = hist / tf.expand_dims(tf.reduce_max(hist, axis=-1), axis=-1)
        return tf.expand_dims(hist, axis=-1)

def _histogram2d(values, value_range, nbins, name='histogram2d'):
    """ histogram2d
    """
    with tf.name_scope(name):
        targets = _digitize(values, value_range, nbins)            # (batch, samples)
        shape = targets.shape

        targets = tf.reshape(targets, [shape[0] * (nbins+1), -1])  # (batch*(nbins+1), samples/(nbins+1))
        max_value = targets.shape[1]  # shape[1] / (nbins+1)

        hist = _bincount(targets, minlength=nbins+1, axis=1)       # (batch*(nbins+1), (nbins+1))
        hist = tf.reshape(hist, (shape[0], nbins+1, -1))           # (batch, (nbins+1), (nbins+1))
        hist = hist / max_value

        return hist

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

def _kl(mu0=0., logvar0=1., mu1=2., logvar1=3.):
    """ Kullback–Leibler divergence
        for mean, logvar

    mu0, sigma0 = norm1
    mu1, sigma1 = norm2

    logvar0 = np.log(np.square(sigma0))
    logvar1 = np.log(np.square(sigma1))

    kld = (np.square(sigma0 / sigma1) + np.square((mu1-mu0)/sigma1) + 2 * np.log(sigma1 / sigma0) - 1) / 2.

    # scale
    # sigma = np.sqrt(np.exp(logvar))

    np.square(sigma0/sigma1) = np.square(sigma0)/np.square(sigma1) = np.exp(logvar0)/np.exp(logvar1) = np.exp(logvar0-logvar1)
    np.square((mu1-mu0)/sigma1) = np.square(mu1-mu0)/np.square(sigma1) = np.square(mu1-mu0)/np.exp(logvar1)
    2 * np.log(sigma1 / sigma0) = np.log(np.square(sigma1 / sigma0)) = np.log(np.exp(logvar1-logvar0)) = logvar1-logvar0

    kld = (np.exp(logvar0-logvar1) + np.square(mu1-mu0)/np.exp(logvar1) + (logvar1 - logvar0) - 1) / 2.
    """
    #return (np.exp(logvar0-logvar1) + np.square(mu1-mu0)/np.exp(logvar1) + (logvar1-logvar0) - 1)/2.
    mask = tf.math.less_equal(logvar0, epsilon)
    logvar0 = logvar0 + tf.cast(mask, dtype=tf.float32) * epsilon
    mask = tf.math.less_equal(logvar1, epsilon)
    logvar1 = logvar1 + tf.cast(mask, dtype=tf.float32) * epsilon
    return (tf.raw_ops.Exp(x=tf.cast(logvar0-logvar1, dtype=tf.float32)) +
            tf.raw_ops.Square(x=tf.cast(mu1-mu0, dtype=tf.float32))/tf.raw_ops.Exp(x=tf.cast(logvar1, dtype=tf.float32)) +
            (logvar1-logvar0) - 1)/2.

def _kld(p=0, q=1):
    """ Kullback–Leibler divergence
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    mask = tf.math.less_equal(q, epsilon)
    q = q + tf.cast(mask, dtype=tf.float32) * epsilon
    mask = tf.math.less_equal(p, epsilon)
    p = p + tf.cast(mask, dtype=tf.float32) * epsilon
    return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)

def _pdf(values, value_range, nbins, name='pdf'):
    """ probability density function
    """
    with tf.name_scope(name):
        targets = _digitize(values, value_range, nbins)
        pdf = _bincount(targets, minlength=nbins+1, axis=1)  # (?,16)
        #pdf = tf.maximum(pdf, 1.0)
        pdf = pdf / tf.expand_dims(tf.reduce_sum(pdf, axis=-1), axis=-1)
        return pdf  # tf.reduce_sum(pdf, axis=-1)

if __name__ == "__main__":

    import numpy as np
    from mylib import print_ndarray

    print('\ntensorflow helper functions')
    print('tensorflow version: {0}'.format(tf.__version__))
    print('tf.executing_eagerly(): {0}'.format(tf.executing_eagerly()))

    # define test random array
    np.random.seed(seed=111)
    mu = np.array([0, 2, 5, -5], dtype=np.float32).reshape(-1,1)
    mu = np.array([0, 0, 0, 0], dtype=np.float32).reshape(-1,1)
    sigma = np.array([1, 2, 3, 3], dtype=np.float32).reshape(-1,1)
    logvar = np.log(np.square(sigma))
    print_ndarray('mu', mu)
    print_ndarray('sigma', sigma)

    a_shape = (4,11000)
    n = a_shape[0]
    a = np.random.normal(size=a_shape) * sigma + mu
    print_ndarray('a = np.random.normal(size={}) * sigma + mu'.format(a.shape), a)

    a = np.cumsum(a, axis=1)
    print_ndarray('a = np.cumsum(a, np.float32)'.format(a.shape), a, frm='8.2f')

    q = [25, 50, 75]
    percentile = np.percentile(a, q, axis=-1)
    print_ndarray('np.percentile(a, {}, axis=-1)'.format(q), percentile)

    percentile = _percentile(a, q, axis=-1)
    percentile = percentile.numpy()
    print_ndarray('_percentile(a, {}, axis=-1)'.format(q), percentile)

    percentile = [25,75]
    treshold = 0.0
    values = a
    print_ndarray('values = np.random.normal(size={}) * sigma + mu'.format(values.shape), values)
    value_range = _value_range(values, percentile=percentile, treshold=treshold)
    value_range = value_range.numpy()
    print_ndarray('value_range = _value_range(values, percentile={}, treshold={})'.format(percentile, treshold), value_range, count=0)
    print_ndarray('', value_range[...,0])

    nbins = 10
    h = _histogram2d(values, value_range, nbins=nbins)
    print_ndarray('h = _histogram2d(values, value_range, nbins={})'.format(nbins), h.numpy())
    print_ndarray('h = np.mean(h, axis=1)', np.mean(h.numpy(), axis=1))

    h = _histogram(values, value_range, nbins=nbins)
    print_ndarray('h = _histogram(values, value_range, nbins={})'.format(nbins), h.numpy()[...,0])

    # test pdf & kl
    print_ndarray('kl(0,1,2,3) = ', _kl().numpy())

    min = tf.reduce_min(value_range)
    max = tf.reduce_max(value_range)
    min, max = (-5., 5.)
    value_range = tf.expand_dims(
            tf.stack(
                [tf.repeat(min, repeats=n),
                tf.repeat(max, repeats=n)],
                axis=0),
            axis=-1
        )
    print_ndarray('value_range', value_range[...,0])

    pdf = _pdf(a, value_range, nbins=nbins)
    print_ndarray('pdf = _pdf(values, value_range, nbins={})'.format(nbins), pdf.numpy())

    p = pdf[0] #tf.stack([pdf[0], pdf[0], pdf[0], pdf[0]], axis=0)
    q = pdf
    print_ndarray('p = pdf[0]', p)
    print_ndarray('q = pdf', q)

    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    kl = _kld(p, q)
    print_ndarray('kl = _kld(p, q)', kl.numpy())


    _mu = np.mean(a, axis=-1)
    _sigma = np.std(a, axis=-1)
    _logvar = np.log(np.var(a, axis=-1))

    print_ndarray('mu ({})'.format(np.reshape(mu, (-1,))), _mu)
    print_ndarray('sigma ({})'.format(np.reshape(sigma, (-1,))), _sigma)
    print_ndarray('logvar ({})'.format(np.reshape(logvar, (-1,))), _logvar)

    kl = _kl(_mu[0], _logvar[0], _mu, _logvar)
    print_ndarray('kl = _kl()', kl.numpy())

    kl = _kl(mu[0], logvar[0], mu, logvar)
    print_ndarray('kl = _kl()', kl.numpy())

    pass # main