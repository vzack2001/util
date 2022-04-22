import numpy as np
import pandas as pd

import functools
from pathlib import Path, WindowsPath


def get_safe_idx(a: np.ndarray, idx=None, data_seq=0, target_seq=0, target_shift=0, dtype=np.int32):
    """ get valid index list for data sequence
            a: np.ndarray, shape of (n,...)
    """
    n = a.shape[0]
    if idx is None:
        idx = np.arange(n, dtype=dtype)

    if isinstance(idx[0], bool):
        idx = np.arange(n, dtype=dtype)[idx]

    idx = np.asarray(idx, dtype=dtype)

    # check data_seq validity
    idx = idx[idx < n]
    idx = idx[idx - (data_seq - 1) >= 0]

    # check target_seq validity
    idx = idx[(idx + (target_seq - 1) + target_shift) < n]
    idx = idx[(idx + target_shift) >= 0]

    #import traceback
    #_stack = traceback.format_stack()
    #[print(i, s.split('\n')[1]) for i, s in enumerate(_stack[:-1])]
    #print('\n')
    return idx

def sequence_from_idx(data: np.ndarray, targets: np.ndarray, idx, data_seq=0, target_seq=0, target_shift=0, dtype=np.float32):
    """ get x, y (data, targets) data sequence for input indices
            as tuple of numpy ndarrays:
                x = data[idx-data_seq:idx]
                y = targets[idx]
    """
    x = []
    y = []
    for i in idx:
        if data_seq > 0:
            x.append(data[i - (data_seq - 1) : i + 1])
        else:
            x.append(data[i])

        if target_seq > 0:
            y.append(targets[i + target_shift : i + target_seq + target_shift])
        else:
            y.append(targets[i + target_shift])

    return np.array(x, dtype=dtype), np.array(y, dtype=dtype) # x, y

class dataset_numpy(object):

    def __init__(self, data, targets=None, data_seq=1, target_seq=0, target_shift=0, dtype=np.float32):

        self.dtype = dtype

        self.data = np.asarray(data, dtype=self.dtype)

        if targets is None:
            self.targets = self.data
            #self.targets = np.copy(self.data)
        else:
            self.targets = np.asarray(targets, dtype=self.dtype)

        assert len(self.data) == len(self.targets), 'targets and data should have same length. Got data[{}] targets[{}]'.format(len(data), len(targets))

        self.set_params(data_seq, target_seq, target_shift)

        self.rec_count  = np.shape(self.data)[0]

        pass  # __init__()

    def __repr__(self):
        s = type(self).__name__
        s += f' (\n\tdata = {type(self.data)}, {np.shape(self.data)} {self.data.dtype} {object.__repr__(self.data)},'
        s += f'\n\ttargets = {type(self.targets)}, {np.shape(self.targets)} {self.data.dtype} {object.__repr__(self.targets)},'
        s += f'\n\trec_count = {self.rec_count},'
        s += f'\n\tdata_seq = {self.data_seq},'
        s += f'\n\ttarget_seq = {self.target_seq},'
        s += f'\n\ttarget_shift = {self.target_shift},'
        shape_x, shape_y = self.shape()
        s += f'\n\tx.shape = {shape_x},'
        s += f'\n\ty.shape = {shape_y},'
        s += f'\n\tdtype = {self.dtype}'
        s += f'\n)'
        return s

    def shape(self, idx=None, data_seq=None, target_seq=None, target_shift=None):
        """ get output x, y shape """

        data_seq, target_seq, target_shift = self.get_params(data_seq, target_seq, target_shift)
        idx = self.safe_idx(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)

        shape_x = [self.size(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)]
        if data_seq > 0:
            shape_x += [data_seq]
        shape_x += list(np.shape(self.data)[1:])

        shape_y = [self.size(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)]
        if target_seq > 0:
            shape_y += [target_seq]
        shape_y += list(np.shape(self.targets)[1:])

        #print(f'.shape(idx={len(idx)}, data_seq={data_seq}, target_seq={target_seq}, target_shift={target_shift}): {tuple(shape_x), tuple(shape_y)}')
        return (tuple(shape_x), tuple(shape_y))

    def set_params(self, data_seq=None, target_seq=None, target_shift=None):
        """ set data & target sequences parameters
        """
        if data_seq is not None:
            self.data_seq = data_seq

        if target_seq is not None:
            self.target_seq = target_seq

        if target_shift is not None:
            self.target_shift = target_shift

        print(f'.set_params(data_seq={self.data_seq}, target_seq={self.target_seq}, target_shift={self.target_shift})')
        return self.data_seq, self.target_seq, self.target_shift

    def get_params(self, data_seq=None, target_seq=None, target_shift=None):
        """ get data & target sequences parameters
        """
        if data_seq is None:
            data_seq = self.data_seq

        if target_seq is None:
            target_seq = self.target_seq

        if target_shift is None:
            target_shift = self.target_shift

        #print(f'.get_params(data_seq={data_seq}, target_seq={target_seq}, target_shift={target_shift})')
        return data_seq, target_seq, target_shift

    def safe_idx(self, idx=None, data_seq=None, target_seq=None, target_shift=None, dtype=np.int32):
        """ get valid index list
        """
        data_seq, target_seq, target_shift = self.get_params(data_seq, target_seq, target_shift)
        idx = get_safe_idx(self.data, idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift, dtype=dtype)
        return idx

    def _get_data(self, idx, data_seq, target_seq, target_shift):
        """ get x, y (data, targets) dataset for input indices
            unsafe to indices
        """
        x, y = sequence_from_idx(self.data, self.targets, idx, data_seq, target_seq, target_shift, self.dtype)
        return x, y

    def get_data(self, idx=None, data_seq=None, target_seq=None, target_shift=None):
        """ get x, y (data, targets) dataset for input indices
        """
        data_seq, target_seq, target_shift = self.get_params(data_seq, target_seq, target_shift)
        idx = self.safe_idx(idx, data_seq, target_seq, target_shift)
        x, y = self._get_data(idx, data_seq, target_seq, target_shift)
        return x, y

    def size(self, **kwarg):
        """ kwarg:
                {idx=None, data_seq=None, target_seq=None, target_shift=None}
        """
        return len(self.safe_idx(**kwarg))

    def shuffle(self, idx=None):
        """ shuffles idx (array in-place)
        """
        idx = self.safe_idx(idx=idx)
        # shuffles the array along the first axis of a multi-dimensional array
        np.random.shuffle(idx)
        return idx

    def split_idx(self, split=(-1,-1), **kwargs):
        """ split 1D idx array to (num_parts, num_steps*n) shape array (n - is natural number) - like (8, 128)
            split_idx(idx=[1 2 3 4 5 6 7 8], split=(3, 4), data_seq=2, target_seq=2, target_shift=2)
            ----
            1   2   3   4
            2   3   4   5
            4   5   6   7
            ----
            kwargs:
                {idx=None, data_seq=None, target_seq=None, target_shift=None, dtype=np.int32}
            return idx - np.ndarray (num_parts, part_steps)
        """
        dtype = kwargs.setdefault('dtype', np.int32)

        safe_idx = self.safe_idx(**kwargs)
        num_parts, num_steps = split

        if num_parts < 0 and num_steps < 0:
            return safe_idx

        size = len(safe_idx)

        if num_parts == -1:
            num_parts = np.int(np.ceil(size/num_steps))

        if num_steps == -1:
            num_steps = np.int(np.ceil(size/num_parts))

        k = num_steps * num_parts

        part_batch = np.int(np.ceil(size/k))
        part_steps = part_batch * num_steps

        # add overlapped samples
        n = part_steps * num_parts - size
        if size < num_steps:
            n = num_steps - size

        if num_parts == 1 or size < num_steps:
            idx = np.arange(size, dtype=dtype)
            mult = num_steps // size
            out = []
            for i in range(num_parts):
                np.random.shuffle(idx)
                _idx = np.concatenate([idx for _ in range(mult)]) if mult > 1 else idx
                _idx = _idx[:n]
                out.append(np.insert(safe_idx, _idx, safe_idx[_idx]))
            return np.asarray(out, dtype=dtype)

        # https://math.stackexchange.com/questions/2975936/split-a-number-into-n-numbers
        # to distribute the number n into p parts, we would calculate the
        # “truncating integer division” of n divided by p, and the corresponding remainder.
        p = num_parts - 1
        d = n // p  # truncating integer division
        r = n % p   # remainder

        # generate parts
        parts = [0] + [d+1 for _ in range(r)] + [d for _ in range(p-r)]

        out = []
        _to = 0
        for i in range(num_parts):
            _from = _to - parts[i]
            _to = _from + part_steps
            idx = safe_idx[_from:_to]
            out.append(idx)

        return np.asarray(out, dtype=dtype)

    def split_batch(self, batch_size=1, **kwarg):
        """ kwarg:
                {idx=None, data_seq=None, target_seq=None, target_shift=None}
        """
        size = self.size(**kwarg)
        split  = (np.int(np.ceil(size/batch_size)), batch_size)
        batches = self.split_idx(split=split, **kwarg)
        return batches  # (?, batch_size)

    def get_sequences(self, idx=None, data_seq=None, target_seq=None, target_shift=None, steps=1):
        """ return idx-started data-targets sequences size of `steps`
            idx = .get_sequences(starts, steps=actor_steps+1)  # (starts=[0, 32, 64, ], steps=32)
            [0,1,2,...,31, 32,33,...,63, 64,65,...,95]
        """
        data_seq, target_seq, target_shift = self.get_params(data_seq, target_seq, target_shift)
        idx = self.safe_idx(idx, data_seq, target_seq, target_shift + steps-1)
        idx = [list(range(start, start + steps)) for start in idx]
        #idx = [np.arange(start=start, stop=start + steps, dtype=idx.dtype) for start in idx]
        idx = np.reshape(idx, -1)
        return idx

    def _on_sequence_start(self, **kwarg):
        pass

    def _gen_sequences(self, idx=None, data_seq=None, target_seq=None, target_shift=None, steps=1, shuffle=False, verbose=False):
        """ return batch iterator (idx-started) data-targets sequences
        """
        data_seq, target_seq, target_shift = self.get_params(data_seq, target_seq, target_shift)
        idx = self.safe_idx(idx, data_seq, target_seq, target_shift + steps-1)
        if shuffle:
            np.random.shuffle(idx)
        self._on_sequence_start(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift, steps=steps, shuffle=shuffle, verbose=verbose)
        for start in idx:
            yield np.arange(start=start, stop=start + steps, dtype=idx.dtype)
        pass  # _gen_sequences()

    def gen_sequences(self, **kwarg):
        """ return batch iterator (idx-started) data-targets sequences
            kwarg:
                {idx=None, data_seq=None, target_seq=None, target_shift=None, steps=1, shuffle=False}
        """
        while True:
            yield from self._gen_sequences(**kwarg)
        pass  # gen_sequences()

    def _on_batch_start(self, **kwarg):
        pass

    def _gen_batch(self, idx=None, data_seq=None, target_seq=None, target_shift=None, batch_size=1, shuffle=False, verbose=False):
        """ generate indices batches
            https://habr.com/ru/post/332074/
            https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
        """
        data_seq, target_seq, target_shift = self.get_params(data_seq, target_seq, target_shift)
        idx = self.safe_idx(idx, data_seq, target_seq, target_shift)
        if shuffle:
            np.random.shuffle(idx)
        batches = self.split_batch(batch_size=batch_size, idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)
        self._on_batch_start(idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift, batch_size=batch_size, shuffle=shuffle, verbose=verbose)
        for i in batches:
            yield i
        pass  # _gen_batch()

    def gen_batch(self, **kwarg):
        """ return batch iterator (idx-started) data-targets sequences
            kwarg:
                {idx=None, data_seq=None, target_seq=None, target_shift=None, batch_size=1, shuffle=False}
        """
        while True:
            yield from self._gen_batch(**kwarg)
        pass  # gen_batch()

    pass  # dataset_numpy(object)


# pd.read_csv helper function
_read_csv = functools.partial(
    pd.read_csv,
    parse_dates=['Date'],
    infer_datetime_format=True,
    header=0,
    index_col=0,
    dtype='float32')

class dataset_pandas(dataset_numpy):

    def __init__(self, data_file=None, target_file=None, data_path='.', target_path=None, data_list=None, targets_list=None, read_fn=_read_csv, **kwargs,):
        """ kwarg
                {data_seq=1, target_seq=0, target_shift=0, dtype=np.float32}
        """
        def _get_dataframe(filename, _file, _path):
            if isinstance(_file, pd.DataFrame):
                df = _file
            elif isinstance(_file, (str, WindowsPath)):
                df = read_fn(_path.joinpath(_file))
            else:
                raise ValueError(f'{filename} is {type(_file)}')
            df['Idx'] = np.arange(df.shape[0], dtype=np.int32)  # add df.index as last column
            return df

        if target_path is None:
            target_path = data_path
        data_path = Path(data_path)
        target_path = Path(target_path)

        self.data_df = _get_dataframe('data_file', data_file, data_path)

        if target_file is None:
            self.targets_df = self.data_df
        else:
            self.targets_df = _get_dataframe('target_file', target_file, target_path)

        self.data_list = data_list or ['Idx']
        self.targets_list = targets_list or ['Idx']

        #self.data_dict = {k:v for v, k in enumerate(self.data_df.columns)}
        #self.targets_dict = {k:v for v, k in enumerate(self.targets_df.columns)}
        self.data_dict = {k:v for v, k in enumerate(self.data_list)}
        self.targets_dict = {k:v for v, k in enumerate(self.targets_list)}

        data = self.data_df[self.data_list].values
        targets = self.targets_df[self.targets_list].values

        super().__init__(data, targets=targets, **kwargs)

        pass  # __init__()

    def __repr__(self):
        super_repr = f'{super().__repr__()}'.split('(\n')
        s = f'{super_repr[0]} ('
        s += f'\n\tdata_df = {self.data_df.shape}, {self.data_df.columns}, {object.__repr__(self.data_df)},'
        s += f'\n\ttargets_df = {self.targets_df.shape}, {self.targets_df.columns}, {object.__repr__(self.targets_df)},'
        s += f'\n\tdata_list = {self.data_list},'
        s += f'\n\ttargets_list = {self.targets_list},'
        s += f'\n\tdata_dict = {self.data_dict},'
        s += f'\n\ttargets_dict = {self.targets_dict},'
        s += f'\n\n{super_repr[1]}'
        return s

    pass  # class dataset_pandas(dataset_numpy)


if __name__ == "__main__":

    from mylib import Profiler, print_ndarray

    with Profiler('test get_safe_idx() & sequence_from_idx()') as p:

        def test_safe_idx_sequence_from_idx(data, idx=None, data_seq=1, target_seq=0, target_shift=0):

            safe_idx = get_safe_idx(data, idx=idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)
            print_ndarray(f'safe_idx = get_safe_idx({np.shape(data)}, idx={idx}, data_seq={data_seq}, target_seq={target_seq}, target_shift={target_shift})', safe_idx)

            x, y = sequence_from_idx(data, data, safe_idx, data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)
            print_ndarray(f'x, _ = sequence_from_idx({np.shape(data)}, {np.shape(data)}, safe_idx, data_seq={data_seq}, target_seq={target_seq}, target_shift={target_shift})', x)
            print_ndarray(f'_, y = sequence_from_idx({np.shape(data)}, {np.shape(data)}, safe_idx, data_seq={data_seq}, target_seq={target_seq}, target_shift={target_shift})', y)

            pass  # test_safe_idx_batch_from_seq()

        size = 12
        data = np.arange(size, dtype=np.float32)
        data = np.reshape(data, (-1,1))
        data = np.concatenate([data + 0.1, data + 0.2], axis=-1)
        print_ndarray('data', data)

        test_safe_idx_sequence_from_idx(data, idx=None, data_seq=0, target_seq=0, target_shift=0)
        test_safe_idx_sequence_from_idx(data, idx=None, data_seq=1, target_seq=0, target_shift=0)
        test_safe_idx_sequence_from_idx(data, idx=None, data_seq=2, target_seq=0, target_shift=0)
        test_safe_idx_sequence_from_idx(data, idx=None, data_seq=2, target_seq=2, target_shift=2)

    with Profiler('test dataset_numpy') as p:
        size = 128
        data = np.arange(size, dtype=np.float32)
        data = np.reshape(data, (-1,1))
        data = np.concatenate([data + 0.1, data + 0.2], axis=-1)
        print_ndarray('\n0 data', data)

        ds_test = dataset_numpy(data, )
        print(ds_test)

        ds_test.set_params(data_seq=2, target_seq=0, target_shift=2)
        print(ds_test)

        safe_idx = ds_test.safe_idx()
        print_ndarray('1 safe_idx = ds_test.safe_idx()', safe_idx)

        safe_idx = ds_test.safe_idx(data_seq=2, target_seq=2, target_shift=2)
        print_ndarray('2 safe_idx = ds_test.safe_idx(data_seq=2, target_seq=2, target_shift=2)', safe_idx)

        safe_idx = ds_test.safe_idx(data_seq=5)
        print_ndarray('3 safe_idx = ds_test.get_safe_idx(data_seq=5)', safe_idx)

        x, y = ds_test.get_data(safe_idx, 5, 2, 2)
        print_ndarray(f'4 x, _ = ds_test.get_data(safe_idx, 5, 2, 2)', x)
        print_ndarray(f'4 _, y = ds_test.get_data(safe_idx, 5, 2, 2)', y)

        #idx = ds_test.split_idx()
        idx = ds_test.split_idx(split=(-1, 18),)
        print_ndarray('5 idx = .split_idx(split=(-1, 18))', idx)

        idx = ds_test.split_idx(split=(3, -1),)
        print_ndarray('5 idx = .split_idx(split=(3, -1))', idx)

        for i in range(1, 6):
            idx = ds_test.split_idx(split=(i, i*3))  # data_seq=2, target_seq=0, target_shift=2
            print_ndarray(f'6/{i} idx = ds_test.split_idx(split={i, i*3})', idx)

        for i in range(1, 6):
            idx = ds_test.split_batch(batch_size=i)
            print_ndarray(f'7/{i} idx = ds_test.split_batch(batch_size={i})', idx)

        safe_idx = ds_test.safe_idx()  #ds_test.shuffle()
        print_ndarray('8 safe_idx = ds_test.safe_idx()', safe_idx)

        for i in range(1, 6):
            idx = ds_test.get_sequences(safe_idx, steps=i)
            print_ndarray(f'9/{i} idx = ds_test.get_sequences(safe_idx, steps={i})', idx, 32, frm='3.0f')

        for i in range(1, 6):
            batch_it = ds_test._gen_sequences(steps=i, shuffle=True)
            print(f'\n10/{i} ds_test._gen_sequences(steps={i}, shuffle=True)')
            for idx in batch_it:
                print(idx)

        print(f'\n11 ds_test.gen_sequences(steps={4}, shuffle=True)')
        batch_it = ds_test.gen_sequences(steps=4, shuffle=True)
        for i, idx in enumerate(batch_it):
            print(f'11/{i}  {idx}')
            if i > 30:
                break

        batch_size = 4
        batch_it = ds_test.gen_batch(batch_size=batch_size, shuffle=True)
        print(f'\n12 ds_test.gen_batch(batch_size={batch_size}, shuffle=True)')
        for i, idx in enumerate(batch_it):
            print(f'12/{i}  {idx}')
            if i > 30:
                break

        pass

    with Profiler('test dataset_pandas') as p:
        size = 12
        data = np.arange(size, dtype=np.float32)
        data = np.reshape(data, (-1,1))
        data = np.concatenate([data + 0.1, data + 0.2], axis=-1)
        print_ndarray('\n0 data', data)

        _cols = ['data1','data2']
        _rows = data
        db = pd.DataFrame(data=_rows, columns=_cols, dtype=np.float32)
        print(db)

        read_csv = functools.partial(
            pd.read_csv,
            infer_datetime_format=True,
            header=0,
            index_col=0,
            dtype='float32')

        data_path = 'D:/Doc/Pyton/util/temp/'
        db.to_csv(data_path + 'temp_data.csv')

        #ds_test = dataset_pandas()  # tested
        #ds_test = dataset_pandas(db, target_file=db)

        ds_test = dataset_pandas('temp_data.csv', target_file=db, data_path=data_path, read_fn=read_csv)
        print(ds_test)

        batch_size = 5
        batch_it = ds_test.gen_batch(batch_size=batch_size, shuffle=True)
        print(f'\n12 ds_test.gen_batch(batch_size={batch_size}, shuffle=True)')
        for i, idx in enumerate(batch_it):
            print(f'12/{i}  {idx}')
            if i > 30:
                break

    pass
