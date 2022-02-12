""" some utills:
    class Profiler(object)
    class data_read_numpy(object)
    def print_ndarray(name, a, count=5, frm='6.2f')
    def ma(x, w=1, divide=True, dtype=np.float32)
    def dma(x, w=1, dtype=np.float32)
    def diff(x, w=1, reversed=False, dtype=np.float32)
"""

import os
import re
import time
import psutil

import functools
from pathlib import Path

import numpy as np
import pandas as pd


# make pd.read_csv helper function
_read_csv = functools.partial(
    pd.read_csv,
    parse_dates=['Date'],
    infer_datetime_format=True,
    header=0,
    index_col=0,
    dtype='float32')

def seq_safe_idx(a: np.ndarray, idx=None, data_seq=1, target_seq=0, target_shift=0):
    """ get valid index list
            a: np.ndarray, shape of (n,...)
    """
    n = a.shape[0]
    if idx is None:
        idx = np.arange(n, dtype=np.int32)

    if isinstance(idx[0], bool):
        idx = np.arange(n, dtype=np.int32)[idx]

    idx = np.asarray(idx, dtype=np.int32)

    # check data_seq validity
    idx = idx[idx < n]
    idx = idx[idx-(data_seq-1) >= 0]

    # check target_seq validity
    idx = idx[(idx+(target_seq-1)+target_shift) < n]
    idx = idx[(idx+target_shift) >= 0]

    return idx

def batch_from_seq(data: np.ndarray, targets: np.ndarray, idx, data_seq=1, target_seq=0, target_shift=0, dtype=np.float32):
    """ get x, y (data, targets) dataset for input indices
            data = x[idx-data_seq:idx]
            targets = y[idx]
        # 0.062 sec (on batch_size=10000 (6589023/6588963) rec. data_seq=60)
    """
    x = []
    y = []
    for i in idx:
        if data_seq > 0:
            x.append(data[i - (data_seq-1) : i + 1])
        else:
            x.append(data[i])
        if target_seq > 0:
            y.append(targets[i + target_shift : i + target_seq + target_shift])
        else:
            y.append(targets[i + target_shift])

    return np.array(x, dtype=dtype), np.array(y, dtype=dtype) # x, y


class data_read_numpy(object):

    def __init__(self, data, targets=None, data_seq=0, target_seq=0, target_shift=0, dtype=np.float32):
        self.dtype = dtype

        self.data = np.asarray(data, dtype=self.dtype)
        if targets is None:
            self.targets = self.data
            #self.targets = np.copy(self.data)
        else:
            self.targets = np.asarray(targets, dtype=self.dtype)

        assert len(self.data) == len(self.targets), 'targets should have same length as data. Got data[{}] targets[{}]'.format(len(data), len(targets))

        self.set_params(data_seq, target_seq, target_shift)

        self.rec_count  = np.shape(self.data)[0]

        pass  # __init__()

    def __str__(self):
        s = type(self).__name__
        s += f' (\n\tdata={type(self.data)}, {np.shape(self.data)} {object.__repr__(self.data)}'
        s += f'\n\ttargets={type(self.targets)}, {np.shape(self.targets)} {object.__repr__(self.targets)}'
        s += f'\n\trec_count={self.rec_count},'
        s += f'\n\tdata_seq={self.data_seq},'
        s += f'\n\ttarget_seq={self.target_seq}'
        s += f'\n\ttarget_shift={self.target_shift},'
        s += f'\n\tdtype={self.dtype})'
        return s

    def len(self, idx=None, data_seq=None, target_seq=None, target_shift=None, batch_size=1):
        """ get safe_idx len
        """
        idx = self.get_safe_idx(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)
        return (len(idx) // batch_size) * batch_size

    def idx_reset(self):
        """ reset index order
        """
        return np.arange(self.rec_count, dtype=np.int32)

    def idx_shuffle(self, idx=None):
        """ shuffles idx (array in-place)
        """
        if idx is None:
            idx = self.idx_reset()
        np.random.shuffle(idx)  # shuffles in-place data indices
        return idx

    def get_overrided(self, data_seq=None, target_seq=None, target_shift=None):
        """ get data & target sequences parameters
        """
        if data_seq is None:
            data_seq = self.data_seq
        if target_seq is None:
            target_seq = self.target_seq
        if target_shift is None:
            target_shift = self.target_shift
        #print('get_overrided(data_seq={}, target_seq={}, target_shift={})'.format(data_seq, target_seq, target_shift))
        return data_seq, target_seq, target_shift

    def set_params(self, data_seq=None, target_seq=None, target_shift=None):
        """ set data & target sequences parameters
        """
        if data_seq is not None:
            self.data_seq = data_seq
        if target_seq is not None:
            self.target_seq = target_seq
        if target_shift is not None:
            self.target_shift = target_shift
        print('set_params(data_seq={}, target_seq={}, target_shift={})'.format(self.data_seq, self.target_seq, self.target_shift))
        return self.data_seq, self.target_seq, self.target_shift

    def _get_data(self, idx, data_seq, target_seq, target_shift):
        """ get x, y (data, targets) dataset for input indices
                data = x[idx-data_seq:idx]
                targets = y[idx]
        print('_get_data(len(data)={}, len(idx)={}, {}, data_seq={}, target_seq={}, target_shift={})'.format(
            self.rec_count, len(idx), idx, data_seq, target_seq, target_shift))
        """

        # 0.062 sec (on batch_size=10000 (6589023/6588963) rec. data_seq=60)
        x = []
        y = []
        for i in idx:
            if data_seq > 0:
                x.append(self.data[i - (data_seq-1) : i + 1])
            else:
                x.append(self.data[i])
            if target_seq > 0:
                y.append(self.targets[i + target_shift : i + target_seq + target_shift])
            else:
                y.append(self.targets[i + target_shift])

        return np.array(x, dtype=self.dtype), np.array(y, dtype=self.dtype) # x, y

    def get_safe_idx(self, idx=None, data_seq=None, target_seq=None, target_shift=None, dtype=np.int32):
        """ get valid index list
        """
        data_seq, target_seq, target_shift = self.get_overrided(data_seq, target_seq, target_shift)

        if idx is None:
            idx = np.arange(self.rec_count, dtype=dtype)

        if isinstance(idx[0], bool):
            idx = np.arange(self.rec_count, dtype=dtype)[idx]

        idx = np.asarray(idx, dtype=dtype)

        # check data_seq validity
        idx = idx[idx < self.rec_count]
        idx = idx[idx-(data_seq-1) >= 0]

        # check target_seq validity
        idx = idx[(idx+(target_seq-1)+target_shift) < self.rec_count]
        idx = idx[(idx+target_shift) >= 0]

        return idx

    def safe_idx_split(self, num_steps=128, num_parts=8, dtype=np.int32, **kwargs):
        """ kwargs
            idx=None, data_seq=None, target_seq=None, target_shift=None
            return idx - np.ndarray (num_parts, part_steps)
        """
        safe_idx = self.get_safe_idx(dtype=dtype, **kwargs)

        size = len(safe_idx)
        k = num_steps * num_parts
        m = size/k

        part_batch = np.int(np.ceil(m))
        part_steps = part_batch * num_steps

        # add overlapped samples
        n = part_steps * num_parts - size

        if num_parts == 1:
            idx = np.arange(size, dtype=dtype)
            np.random.shuffle(idx)
            idx = idx[:n]
            out = np.insert(safe_idx, idx, safe_idx[idx])
            return np.reshape(out, (1,-1))

        # https://math.stackexchange.com/questions/2975936/split-a-number-into-n-numbers
        # to distribute the number n into p parts, we would calculate the
        # “truncating integer division” of n divided by p, and the corresponding remainder.
        p = num_parts - 1
        d = n // p  # truncating integer division
        r = n % p   # remainder

        # generate parts
        parts = [0] + [d+1 for i in range(r)] + [d for i in range(p-r)]

        out = []
        _to = 0
        for i in range(num_parts):
            _from = _to - parts[i]
            _to = _from + part_steps
            idx = safe_idx[_from:_to]
            out.append(idx)

        return np.asarray(out, dtype=dtype)

    def get_dataset(self, idx=None, data_seq=None, target_seq=None, target_shift=None):
        """ return whole data-targets sequences
        """
        data_seq, target_seq, target_shift = self.get_overrided(data_seq, target_seq, target_shift)
        idx = self.get_safe_idx(idx, data_seq, target_seq, target_shift)
        return self._get_data(idx, data_seq, target_seq, target_shift)  # x, y

    def get_batch_idx(self, idx=None, data_seq=None, target_seq=None, target_shift=None, batch_size=1):
        """ return batch (idx-started) data-targets sequences
            idx = .get_batch_idx(starts, batch_size=actor_steps+1)  # starts=[0, 31, 63, ] batch_size=32
        """
        data_seq, target_seq, target_shift = self.get_overrided(data_seq, target_seq, target_shift)
        idx = self.get_safe_idx(idx, data_seq, target_seq, target_shift)
        idx = [list(range(i, i + batch_size)) for i in idx]  # (len(idx), batch_size)
        idx = np.reshape(idx, -1)
        return idx  # self._get_data(idx, data_seq, target_seq, target_shift)  # x, y

    def _gen_batch(self, idx, data_seq, target_seq, target_shift, batch_size, shuffle):
        """ generate x, y batches unsafe to batch_size
            https://habr.com/ru/post/332074/
            https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
        """
        if shuffle:
            np.random.shuffle(idx)
        idx = np.reshape(idx, (-1, batch_size))
        batch_count = np.shape(idx)[0]
        for i in range(batch_count):
            x, y = self._get_data(idx[i], data_seq, target_seq, target_shift)
            yield x, y

        pass  # _gen_batch()

    def gen_batch_idx(self, idx, data_seq=None, target_seq=None, target_shift=None, batch_size=1, shuffle=False, verbose=False):
        """ generate x, y batches
            align on batch_size
        """
        #print('gen_batch_idx(')

        data_seq, target_seq, target_shift = self.get_overrided(data_seq, target_seq, target_shift)
        idx = self.get_safe_idx(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)

        if verbose:
            print('gen_batch(len(idx)={}/{}, data_seq={}, target_seq={}, target_shift={}, batch_size={}, shuffle={})'.format(
                   len(idx), len(idx) // batch_size, data_seq, target_seq, target_shift, batch_size, shuffle))

        # align idx on batch_size
        start_from = len(idx) - (len(idx) // batch_size) * batch_size
        idx = idx[start_from:]

        yield from self._gen_batch(idx, data_seq, target_seq, target_shift, batch_size, shuffle)

        #print('gen_batch() the end')
        pass  # gen_batch_idx()

    def gen_batch(self, data_seq=None, target_seq=None, target_shift=None, batch_size=1, shuffle=False, verbose=False):
        """ generate x, y batches
            align on batch_size
        """
        #print('gen_batch(')
        data_seq, target_seq, target_shift = self.get_overrided(data_seq, target_seq, target_shift)
        idx = self.get_safe_idx(idx=None, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)

        if verbose:
            print('gen_batch(len(idx)={}/{}, data_seq={}, target_seq={}, target_shift={}, batch_size={}, shuffle={})'.format(
                   len(idx), len(idx) // batch_size, data_seq, target_seq, target_shift, batch_size, shuffle))

        # align idx on batch_size
        start_from = len(idx) - (len(idx) // batch_size) * batch_size
        idx = idx[start_from:]

        yield from self._gen_batch(idx, data_seq, target_seq, target_shift, batch_size, shuffle)

        #print('gen_batch() the end')
        pass  # gen_batch()

    pass  # data_read_numpy


class data_read_pandas(data_read_numpy):

    def __init__(self, data_file=None, target_file=None, data_path=None, data_list=None, targets_list=None, read_fn=None, **kwargs,):

        #assert datafile is None and targetfile is None, 'targets should have same length as data. Got data[{}] targets[{}]'.format(len(data), len(targets))
        if data_file is None and target_file is None:
            raise ValueError(
                """data_file is None and target_file is None"""
            )

        data_path = data_path or '.'
        data_path = Path(data_path)
        #print(data_path.absolute())

        data_file = data_path.joinpath(data_file)

        #self.read_fn = _read_csv if read_fn is None else read_fn
        if read_fn is None:
            read_fn = _read_csv

        with Profiler('pd.read_csv({})'.format(data_file)) as p:
            self.data_df = read_fn(data_file)
            self.data_df['Idx'] = np.arange(self.data_df.shape[0], dtype=np.int32)  # self.data_df.index
            #print(self.data_df.head())
            #print(self.data_df.info())
            print('data_df.shape={}'.format(self.data_df.shape))
            print('data_df.columns=\n{}'.format(self.data_df.columns))

        if target_file is None:
            self.target_df = self.data_df
        else:
            target_file = data_path.joinpath(target_file)
            with Profiler('pd.read_csv({})'.format(target_file)) as p:
                self.target_df = read_fn(target_file)
                self.target_df['Idx'] = np.arange(self.target_df.shape[0], dtype=np.int32)  # self.target_df.index
                print('target_df.shape={}'.format(self.target_df.shape))
                print('target_df.columns=\n{}'.format(self.target_df.columns))

        self.data_list = data_list or ['Idx']
        print('data_list', self.data_list)

        self.targets_list = targets_list or ['Idx']
        print('targets_list', self.targets_list)

        self.data_dict =  {k:v for v,k in enumerate(self.data_list)}
        self.targets_dict =  {k:v for v,k in enumerate(self.targets_list)}

        data = self.data_df[self.data_list].values
        targets = self.target_df[self.targets_list].values

        super().__init__(data, targets=targets, **kwargs)

        pass  # __init__()

    pass  # data_read_pandas


class Profiler(object):
    def __init__(self, name='Profiler', expected_time=None, awaited_time=None):
        expected_time = expected_time or awaited_time
        del awaited_time
        self.name = name
        self.expected_time = expected_time

    def __enter__(self):
        s = self.name
        if self.expected_time is not None:
            s += ', ET: {:.1f} sec.'.format(self.expected_time)
        print('\n>>>', s, '>>>')
        self._startTime = time.time()
        self._stepTime = self._startTime
        self._pid = os.getpid()
        self._py  = psutil.Process(self._pid)
        self._startMem = self._py.memory_info()
        return self

    def __exit__(self, type, value, traceback):
        self._mem = self._py.memory_info()
        startTime = time.time() - self._startTime
        #datetime.timedelta(seconds=startTime)
        print('<<<', self.name, '<<<', ' ET: {:.3f} sec.'.format(startTime),
                                       ' Mem: rss = {:.2f} MB'.format(self._mem[0]/2.**20),
                                       '({:.2f} MB)'.format((self._mem[0]-self._startMem[0])/2.**20),
                                       '/ vms = {:.2f} MB'.format(self._mem[1]/2.**20),
                                       '({:.2f} MB)'.format((self._mem[1]-self._startMem[1])/2.**20), '\n')
    def __repr__(self):

        def get_time_str(seconds, ticks=False):
            time_str = 'n/a'
            if seconds < 120:
                frm = '.1f'
                if seconds < 100:
                    frm = '.2f'
                    if seconds < 10:
                        frm = '.3f'
                time_str = f'{seconds:{frm}}' + (' sec' if ticks else '')
            else:
                frm = '%H:%M' + (' h:m' if ticks else '')
                if seconds < 3600:
                    frm = '%M:%S' + (' m:s' if ticks else '')
                time_str = time.strftime(frm, time.gmtime(seconds))

            return time_str

        stepTime = time.time() - self._stepTime
        startTime = time.time() - self._startTime
        s = '{}/{}'.format(get_time_str(stepTime), get_time_str(startTime, ticks=True))
        if self.expected_time is not None:
            s += ' {:5.1f}%'.format((startTime)/self.expected_time * 100.)
        self._stepTime = time.time()
        return s

    pass  # class Profiler(object)


def _is_empty(a):
    if a is None:
        return True
    try:
        if np.prod(np.shape(a)) == 0:
            return True
    except:
        return None
    return False


def _data_format_string(a):
    # set data format string
    # '{}.{}f'.format(max_len, fract_part)
    # https://habr.com/ru/post/112953/  "про арифметику с плавающей запятой"
    # ----------------------------

    format_string = '11.4e'  # -2.1234e+02  m +7
    a = np.asarray(a)

    # - 1 for sign, -1 for decimal point
    # for p =     0, 1, 2, 3, 4 -2 -1
    fract_part = [4, 3, 2, 1, 1, 5, 5]

    s = 0  # sign
    m = 0  # mantissa
    p = 0  # and exponent

    a_min  = np.min(a)
    s = 1 if a_min < 0 else 0
    # for my will %)
    s = 1

    a_max = max(np.abs(np.max(a)), np.abs(a_min))
    p = np.int32(np.ceil(np.log10(a_max))) if a_max > 0 else 0

    if p > 7 or p < -2:
        return format_string  #'11.4e'

    m = 0 if p > 4 else fract_part[p]
    p = max(p, 1)

    if type(a.flat[0]).__name__.find('int') > -1:
        m = 0

    format_string = '{}.{}f'.format(s + p + 1 + m, m)

    return format_string


def print_ndarray(name, a, count=12, frm=None, with_end=True, p1=10, p99=90):
    #print('print_ndarray -----------------------------------------------------------------------')
    #print('name: {}\ntype(a): {}\nnp.shape(a): {}\ncount: {}\nfrm: {}\nwith_end: {}'.format(name, type(a), np.shape(a), count, frm, with_end))
    #print('np.shape(a): {}\n'.format(np.shape(a)))

    type_a = object.__repr__(a)  #type(a)
    shape_a = np.shape(a)
    a_is_empty = _is_empty(a)
    if a_is_empty is None:
        return None

    a = np.asarray(a)

    if len(a.flat) == 0:
        type_a_flat = type(None)
    else:
        type_a_flat = type(a.flat[0])

    def get_header_str(name, a):
        h = ''
        f = ''
        if len(name) > 0:

            if name[0] == '\n':
                name = name[1:]
                h += '\n'

            f += '----'
            if name[-1] == '\n':
                name = name[:-1]
                f += '\n'

            h += '----\n'
            h += '{} {}'.format(name, type_a)
            h += ' {}'.format(shape_a)
            #h += ' {} bytes'.format(sys.getsizeof(a))

            if len(a.flat) > 0:
                h += ' {}'.format(type_a_flat)

        return h, f

    if a_is_empty:
        frm = ''

    if frm is None:
        frm = _data_format_string(a)

    header_str, footer_str = get_header_str(name, a)

    def get_stat_str(a):
        # print array stats
        s =  ' ({:{frm}}/{:{frm}}'.format(np.mean(a), np.mean(np.abs(a)), frm=re.sub('.*\.', '.', _data_format_string(np.mean(np.abs(a)))))
        s += ', {:{frm}}/{:{frm}}'.format(np.std(a), np.std(np.abs(a)),   frm=re.sub('.*\.', '.', _data_format_string(np.std(a))))
        s += ', [{:{frm}}/{:{frm}}]'.format(np.min(a), np.max(a),         frm=re.sub('.*\.', '.', frm))
        s += ' p{}/{}={:{frm}}/{:{frm}})'.format(p1, p99, np.percentile(a, p1), np.percentile(a, p99), frm=re.sub('.*\.', '.', frm))
        return s

    def get_body_str(a):

        # ??? convert to 2D-array or # raise ValueError('Array shape size must be 2 or less. Try some array slice..')
        if len(np.shape(a)) == 4:
            a = a[0,:,:,0]
        if len(np.shape(a)) == 3:
            a = a[0,:,:]
        if len(np.shape(a)) > 4:
            a = a.flat
        # convert to 2D-array
        if len(np.shape(a)) < 2:
            a = np.reshape(a, (1, -1))
        # Now we have 2D ndarray

        # format string for delimiter
        frm_str = '{:{frm}}'.format(a.flat[0], frm=frm)
        frm_str_len = len(frm_str)
        delimiter_frm = '>{}s'.format(frm_str_len)

        rows = 0
        cols = 0
        if isinstance(count, int):               # count=5
            rows = min(count, np.shape(a)[0])
            cols = min(count, np.shape(a)[1])
        if isinstance(count, tuple):             # count=(5,10)
            rows = min(count[0], np.shape(a)[0])
            cols = min(count[1], np.shape(a)[1])

        if rows * cols == 0:
            return ''

        rows_all = True
        cols_all = True
        if rows < np.shape(a)[0]:
            rows_all = False
        if cols < np.shape(a)[1]:
            cols_all = False

        rows_list = [[i for i in range(0, rows)]]
        cols_list = [slice(0,cols)]

        if with_end:
            rows_all = True
            if rows < np.shape(a)[0]:
                rows = max(rows // 2, 1)
                rows_list = [[i for i in range(0, rows)], [i for i in range(-rows,0)]]
            cols_all = True
            if cols < np.shape(a)[1]:
                cols = max(cols // 2, 1)
                cols_list = [slice(0,cols), slice(-cols, None)]

        s = ''
        for rows in rows_list:
            for row in rows:
                for col_slice in cols_list:
                    s += ' '.join(['{:{frm}}'.format(x, frm=frm) for x in a[row, col_slice]])
                    s += '{:{frm}}'.format('...', frm=delimiter_frm)
                    #s += ' ... '
                if cols_all:
                    s = s[:-frm_str_len]
                s += '\n'

            s += '{:{frm}}\n'.format('...', frm=delimiter_frm)
        if rows_all:
            s = s[:-frm_str_len-1]
        return s[:-1]

    if len(header_str) > 0:
        print(header_str)
        if not a_is_empty:
            stat_str = get_stat_str(a)
            print(stat_str)
        print('----')

    body_str = ''
    if not a_is_empty:
        body_str = get_body_str(a)
        print(body_str)

    if len(footer_str) > 0 and len(body_str) > 0:
        print(footer_str)

    pass  # print_ndarray()


def what_is(name, x, methods=True, iterables=False):
    print('--------')
    print(name, type(x))
    print('methods={}, iterables={}'.format(methods, iterables))
    print('--------')
    print(x)

    if isinstance(x, list):
        print('\n'.join(['%s' % type(item) for item in x]))
        print('--------')

    items = dir(x)
    names = []  # [str(item) for item in items]
    descr = []
    iters = []

    if methods is not False:
        print('--------')
        #print('\n'.join(['%s' % item for item in dir(x)]))
        #print('try find iterable -------------')
        for item in items:
            s = str(item)
            names.append(s)
            try:
                item_eval = eval('x.' + s) #!!! warning !!! use 'eval' function !!!
            except Exception as e:
                s += "<Exception '" + str(e) + "'>"
                #print('Exception', e)
                descr.append(str(e))
            else:
                #if s.find('__') < 0:
                #descr.append(str(type(item_eval)))
                s += ' ' + str(type(item_eval))
                if isinstance(item_eval, (bool, float, int, str)):
                    s += ' ' + str(item_eval)
                if isinstance(item_eval, list):
                    iters.append(str(item) + ' ' + str(item_eval))
            descr.append(s)
            #print(s)
    if iterables is not False:
        print('--------')
        print('\n'.join(['%s' % it for it in iters]))
    print('--------')
    #print('\n'.join(['%s' % name for name in names if name.find('__', 0) == -1]))
    print('\n'.join(['%s' % name for name in descr]))
    print('--------')
    pass  # what_is()


def ma(x, w=1, divide=True, dtype=np.float32):
    """ calculate moving average using numpy
    """
    assert w > 0, 'w - must be greater or equal 1'
    a = np.ones(w)
    b = np.convolve(x, a, mode='full')
    if w > 1:
        b = b[:-w+1]
    if divide:
        b = b/w
    return np.asarray(b, dtype=dtype)

def dma(x, w=1, dtype=np.float32):
    """ calculate (x - ma(w))
    """
    assert w > 0, 'w - must be greater or equal 1'
    a = np.arange(w-1, -1, -1, dtype=np.float32)/w
    b = np.convolve(x, a, mode='full')
    if w > 1:
        b = b[:-w+1]
    return np.asarray(b, dtype=dtype)

def diff(x, w=1, reversed=False, dtype=np.float32):
    """ calculate difference b[i] = x[i] - x[i-w]
        if reversed=True :   b[i] = x[i+w] - x[i]
    """
    w += 1
    assert w > 0, 'w - must be greater or equal 1'
    if w == 1:
        return np.zeros_like(x)

    a = np.zeros(w, dtype=np.float32)
    a[0] = 1
    a[-1] = -1

    b = np.convolve(x, a, mode='full')

    if reversed:
        b[-w+1:] = 0
        return np.asarray(b[w-1:], dtype=dtype)
    else:
        b[:w-1] = 0
        return np.asarray(b[:-w+1], dtype=dtype)


# test
if __name__ == "__main__":

    import sys
    print('\npython helper functions')
    print('\npython version: {0}'.format(sys.version))
    print('numpy version: {0}'.format(np.__version__))

    np.random.seed(seed=111)

    with Profiler('mylib.py testing', expected_time=0.042) as p:

        print('\nprint_ndarray(name, a, count=5, frm="6.2f", with_end=False)')

        a = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float32)
        print('\na = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float32)')

        print_ndarray('1: (a, count=10, with_end=False)', a, count=10, frm='6.0f')
        print(p)

        print_ndarray('2: (a, count=5, with_end=False)', a, count=5, frm='6.0f', with_end=False)
        print(p)

        print_ndarray('3: (a, count=5, with_end=True)', a, count=5, with_end=True)
        print(p)


        a = np.arange(0, 120, dtype=np.int32)
        print('\na = np.arange(0, 120, dtype=np.float32)')
        a = np.reshape(a, (10, 12))
        print('a = np.reshape(a, (30, 40))')

        print_ndarray('4: (a, count=10, with_end=False)', a, count=(6,10))
        print(p)

        print_ndarray('5: (a, count=(6, 10), with_end=True)', a, count=(6,10), with_end=True)
        print(p)

        print_ndarray('6: (a, count=(20, 20), with_end=True)', a, count=(20,20), with_end=False)
        print(p)

        print_ndarray('7: (a, count=(20, 20), with_end=True)', a, count=(20,20), with_end=True)
        print(p)

        a = np.reshape(a, (-1, 10, 12, 1))
        print('\na = np.reshape(a, (-1, 10, 12))')
        try:
            print_ndarray('8: (a, count=(20, 20), with_end=True)', a, count=(20,20), with_end=True)
        except ValueError:
            print('Wrong dim size:', ValueError)
        print(p)


        np.random.seed(seed=111)

        a = np.random.normal(0, 1, size=(100, 100))
        print('\na = np.random.normal(0, 1, size=(100, 100))')
        print_ndarray('9: (a, count=(5, 10), with_end=True)', a, count=(5,10), with_end=True)
        print(p)

        a = np.random.normal(100, 1, size=(100, 100))
        print('\na = np.random.normal(100, 1, size=(100, 100))')
        print_ndarray('9: (a, count=(5, 10), with_end=True)', a, count=(5,10), with_end=True)
        print(p)

        '''
        a = np.random.normal(0, 0.01, size=(100, 100))
        print('\na = np.random.normal(0, 0.01, size=(100, 100))')
        print_ndarray('9: (a, count=(5, 10), with_end=True)', a, count=(5,10), with_end=True)
        print(p)

        a = np.random.normal(0, 0.0001, size=(100, 100))
        print('\na = np.random.normal(0, 0.001, size=(100, 100))')
        print_ndarray('9: (a, count=(5, 10), with_end=True)', a, count=(5,10), with_end=True)
        print(p)
        '''
        print('\n-------------------------------')
        print('def ma(x, w):')
        a = np.arange(12, dtype=np.float32)
        print_ndarray('a = np.arange(12)', a, count=20, frm='7.0f')
        print_ndarray('ma(a, 3)', ma(a, 3), count=20)
        print_ndarray('ma(a, 3, divide=False)', ma(a, 3, divide=False), count=20)

        print('\n-------------------------------')
        print('def dma(x, w):')
        x = np.asarray([1, 2, 4, 7, 11, 16, 22, 29, 37, 46], dtype=np.float32)
        #                  1  2  3   4   5   6   7   8   9
        print_ndarray('', x)

        y = diff(x, 1)
        print_ndarray('y = diff(x, 1)', y)
        #print_ndarray('ma(y, 3)', ma(y, 3, divide=False)) #

        w = 3
        a = ma(x, w)
        print_ndarray('a = ma(x, {})'.format(w), a)

        xa = x - a
        print_ndarray('x - a', xa)
        print_ndarray('dma(y, {})'.format(w), dma(y, w))

    print('data_read_pandas ====================================================================')
    filename = 'data/cdata_test.csv'
    with Profiler('df = pd.read_csv({}, header=0, index_col=0)'.format(filename)) as p:
        df = pd.read_csv(filename, dtype='float32', parse_dates=['Date'], infer_datetime_format=True, header=0, index_col=0)
        print(df.columns)
        # Index(['Date','1st','2nd','3rd','4th'], dtype='object')

    data_list = ['1st', '2nd', '3rd']
    targets_list=['4th']

    # (data, targets=None, data_seq=0, target_seq=0, target_shift=0, dtype=np.float32)
    dr = data_read_numpy(df[data_list].values, targets=df[targets_list].values)
    dr.set_params(data_seq=4, target_seq=2, target_shift=1)
    print(dr)
    print('=================================================================================')

    print('x, y = dr.get_dataset()')
    x, y = dr.get_dataset()
    print_ndarray('x, _ = dr.get_dataset()', x)
    print_ndarray('_, y = dr.get_dataset()', y)
    print('=================================================================================')

    n = 867996
    actor_steps = 128

    a = np.asarray(range(n), dtype=np.float32)
    print_ndarray(f'a = np.asarray(range({n}), dtype=np.float32)', a, frm='6.0f')

    dr = data_read_numpy(a, targets=a, data_seq=60, target_seq=0, target_shift=0)
    print(dr)
    print('dr.data[-1] =', dr.data[-1])

    print('\ndr.len():', dr.len())
    print('dr.len(data_seq=10, target_seq=0, target_shift=0):', dr.len(data_seq=10, target_seq=0, target_shift=0))
    print('dr.len(batch_size=10):', dr.len(batch_size=10))
    print()

    idx = dr.get_safe_idx(data_seq=60, target_seq=1, target_shift=1)
    print_ndarray('idx = dr.get_safe_idx(data_seq=60, target_seq=1, target_shift=1)', idx)

    idx = dr.safe_idx_split(num_steps=actor_steps, num_parts=8, data_seq=60, target_seq=1, target_shift=1)
    print_ndarray(f'idx = dr.safe_idx_split(num_steps={actor_steps}, num_parts=8, data_seq=60, target_seq=1, target_shift=1)', idx)

    idx = idx[:,::actor_steps]       # (8, 848) # start positions
    print_ndarray(f'idx = idx[:,::{actor_steps}]', idx)

    idx = [idx[0,0], idx[-1,-1]]
    print('[idx[0,0], idx[-1,-1]]', idx)

    idx = dr.get_batch_idx(idx, batch_size=actor_steps+1)
    print('dr.get_batch_idx(idx, batch_size=actor_steps):', idx, np.shape(idx))

    x, y = dr.get_dataset(idx)
    print_ndarray(f'x,_ = dr.get_dataset({len(idx)})', x)
    print_ndarray(f'_,y = dr.get_dataset({len(idx)})', y)

    #idx = np.reshape(idx, (-1, actor_steps+1))


    import traceback
    a = []
    print('\na = []')
    try:
        print_ndarray('10: a = []', a, count=(5,10), with_end=True)
    except Exception as e:
        #print(e)
        traceback.print_exc()

    a = None
    print('\na = None')
    try:
        print_ndarray('11: a = None', a, count=(5,10), with_end=True)
    except Exception as e:
        traceback.print_exc()

    a = np.random.normal(0, 1, size=(100, 100))
    print('\na = np.random.normal(0, 1, size=(100, 100))')
    print_ndarray('9: (a, count=(5, 10), with_end=True)', a, count=0, with_end=True)


    with Profiler('Profiler testing', expected_time=100) as p:
        for i in range(20):
            print(p)
            time.sleep(i)

    pass  # test section
