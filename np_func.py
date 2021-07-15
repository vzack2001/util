import numpy as np

epsilon = 1e-16

def _value_range(values, percentile=[10,90], treshold=0):
    """ input
            values (batch,seq)
        return
            value_range (2,batch,1)
    """
    dtype = np.dtype(values.flat[0])                                     # np.float32
    value_percentile = np.percentile(values, percentile, axis=-1)        # (2,batch) /2 -- [min, max]/
    value_last = values[:,-1]                                            # (batch,)
    value_range = np.stack([
        np.minimum(value_percentile[0], value_last - treshold),          # (2,batch) /2 -- [min, max]/
        np.maximum(value_percentile[1], value_last + treshold)
        ], axis=0)
    value_range = np.asarray(value_range, dtype=dtype)
    return value_range    # (2,?)

def _hist_bin(value_range, nbins):
    """ multidimentional analog numpy histogram_bin_edges():
        input
            value_range (2,batch)
            nbins
        return
            bins (nbins, batch)
    """
    bins = np.linspace(start=value_range[0], stop=value_range[1], num=nbins)
    return bins

def _digitize(values, value_range, nbins, dtype=np.int32):
    """ multidimentional analog numpy digitize(), work exactly like:
            bins = np.linspace(start=value_range[0], stop=value_range[1], num=nbins)
            indices = np.digitize(values, bins)
        input
            values: array shape of (batch, samples)
            value_range: array shape of (2, batch), value_range[0]=min, value_range[1]=max
            nbins: number of bins, see example
        return indices:
            array shape of (batch, samples)
    """
    value_range = np.expand_dims(value_range, axis=-1)
    scaled_values = (values - value_range[0]) / (value_range[1] - value_range[0])
    indices = np.floor(scaled_values * (nbins-1))
    indices = np.clip(indices, -1, nbins-1) + 1
    return np.asarray(indices, dtype=dtype)

def _bincount(targets, minlength=None, axis=None, dtype=np.float32):
    """ multidimentional analog numpy bincount(), work exactly like:
            bincount(targets, minlength=nclass)
        targets: array shape of (batch, samples) of binned values indices
        minlength: number of classes
        return Count_numbers array shape of (batch,nclass)
    """
    if minlength is None:
        minlength = np.max(targets)
    return np.asarray(np.sum(_one_hot(targets, minlength, dtype=np.int8), axis=axis), dtype=dtype)

def _one_hot(targets, nclass, dtype=np.float32):
    """ one_hot
        targets: array shape of (batch, samples) of binned values indices
        nclass: number of classes
        return one_hot array shape of (batch,samples,nclass)
    """
    #return np.array(nclass[..., None] == np.arange(nclass), dtype)
    return np.eye(nclass, dtype=dtype)[targets]

def _histogram(values, value_range, nbins):
    """ histogram
        input
            values: array shape of (batch, samples)
            value_range: array shape of (2, batch), value_range[0]=min, value_range[1]=max
            nbins: number of bins, see example
        return
            array shape of (batch, nbins+1)
    """
    targets = _digitize(values, value_range, nbins)       # (batch, samples)
    hist = _bincount(targets, minlength=nbins+1, axis=1)  # (batch, nbins+1)

    max_value = targets.shape[1]
    hist = hist / max_value

    #hist = hist / np.expand_dims(np.max(hist, axis=-1), axis=-1)
    #hist = hist / np.expand_dims(np.sum(hist, axis=-1), axis=-1)
    return hist  # np.expand_dims(hist, axis=-1)

def _histogram2d(values, value_range, nbins):
    """ histogram2d
        input
            values: array shape of (batch, samples)
            value_range: array shape of (2, batch), value_range[0]=min, value_range[1]=max
            nbins: number of bins, see example
        return
            array shape of (batch, nbins+1, nbins+1)
    """
    targets = _digitize(values, value_range, nbins)            # (batch, samples)
    shape = targets.shape

    targets = np.reshape(targets, (shape[0] * (nbins+1), -1))  # (batch*(nbins+1), samples/(nbins+1))
    max_value = targets.shape[1]

    hist = _bincount(targets, minlength=nbins+1, axis=1)       # (batch*(nbins+1), (nbins+1))
    hist = np.reshape(hist, (shape[0], nbins+1, -1))           # (batch, (nbins+1), (nbins+1))
    hist = hist / max_value

    return hist

if __name__ == "__main__":

    import numpy as np
    from mylib import print_ndarray

    print('\nnumpy helper functions')
    print('numpy version: {0}'.format(np.__version__))

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
    a = np.asarray(a, dtype=np.float32)
    print_ndarray('a = np.random.normal(size={}) * sigma + mu'.format(a.shape), a)

    a = np.cumsum(a, axis=1)
    print_ndarray('a = np.cumsum(a, np.float32)'.format(a.shape), a, frm='8.2f')

    q = [25, 50, 75]
    percentile = np.percentile(a, q, axis=-1)
    print_ndarray('np.percentile(a, {}, axis=-1)'.format(q), percentile)

    percentile = [25,75]
    treshold = 0.0
    values = a
    print_ndarray('values = np.random.normal(size={}) * sigma + mu'.format(values.shape), values)

    value_range = _value_range(values, percentile=percentile, treshold=treshold)
    print_ndarray('1) value_range = _value_range(values, percentile={}, treshold={})'.format(percentile, treshold), value_range)

    nbins = 10

    bins = _hist_bin(value_range, nbins)
    print_ndarray('bins = _hist_bin(value_range, nbins)', bins)

    h = _histogram2d(values, value_range, nbins)
    print_ndarray('h = _histogram2d(values, value_range, nbins={})'.format(nbins), h)
    print_ndarray('h = np.mean(h, axis=1)', np.mean(h, axis=1))

    h = _histogram(values, value_range, nbins)
    print_ndarray('h = _histogram(values, value_range, nbins={})'.format(nbins), h) #[...,0]
    print_ndarray('np.sum(h, axis=1)', np.sum(h, axis=1)) #[...,0]
