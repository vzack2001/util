""" some utills:
    class Profiler(object)
    def print_ndarray(name, a, count=5, frm='6.2f')
"""

import os
import re
import sys
import time
import psutil
import numpy as np


class Profiler(object):
    def __init__(self, name='Profiler'):
        self.name = name

    def __enter__(self):
        print('\n>>>', self.name, '>>>')
        self._startTime = time.time()
        self._stepTime = self._startTime
        self._pid = os.getpid()
        self._py  = psutil.Process(self._pid)
        self._startMem = self._py.memory_info()
        return self
        
    def __exit__(self, type, value, traceback):
        self._mem = self._py.memory_info()
        print('<<<', self.name, '<<<', ' ET: {:.3f} sec.'.format(time.time() - self._startTime), 
                                       ' Mem: rss = {:.2f} MB'.format(self._mem[0]/2.**20), 
                                       '({:.2f} MB)'.format((self._mem[0]-self._startMem[0])/2.**20), 
                                       '/ vms = {:.2f} MB'.format(self._mem[1]/2.**20), 
                                       '({:.2f} MB)'.format((self._mem[1]-self._startMem[1])/2.**20), '\n')
    def __repr__(self):
        s = '{:.3f} sec.'.format(time.time() - self._stepTime)
        self._stepTime = time.time()
        return s

    pass  # class Profiler(object)


def _data_format_string(a):
    # set data format string
    # '{}.{}f'.format(max_len, fract_part)
    # https://habr.com/ru/post/112953/  "про арифметику с плавающей запятой"
    # ----------------------------

    format_string = '11.4e'  # -2.1234e+02  m +7

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
    p = np.int32(np.ceil(np.log10(a_max))) if a_max > 0 else -10      # just for remember (no matter in this case)
    #p = np.where(a_max > 0, np.int32(np.ceil(np.log10(a_max))), -10) # in LARGE arrays np.where is MUCH faster
                                                                      # BUT it's calculate BOTH conditionals values
    if p > 7 or p < -2:
        return format_string  #'11.4e'

    m = 0 if p > 4 else fract_part[p]
    p = max(p, 1)

    if type(a.flat[0]).__name__.find('int') > -1:
        m = 0

    format_string = '{}.{}f'.format(s + p + 1 + m, m)

    return format_string


def print_ndarray(name, a, count=5, frm=None, with_end=True, p1=10, p99=90):
    #print('print_ndarray -----------------------------------------------------------------------')
    #print('name: {}\ntype(a): {}\nnp.shape(a): {}\ncount: {}\nfrm: {}\nwith_end: {}'.format(name, type(a), np.shape(a), count, frm, with_end))
    #print('np.shape(a): {}\n'.format(np.shape(a)))

    s = '{} {}'.format(name, type(a))
    a = np.asarray(a)
    s += ' {}'.format(np.shape(a))
    #s += ' {} bytes'.format(sys.getsizeof(a))

    if frm is None:
        frm = _data_format_string(a)

    # ??? convert to 2D-array or # raise ValueError('Array shape size must be 2 or less. Try some array slice..')
    if len(np.shape(a)) == 3:
        a = a[0,:,:]
    if len(np.shape(a)) > 3:
        a = a.flat
    # convert to 2D-array
    if len(np.shape(a)) <= 1:
        a = np.reshape(a, (1, -1))

    # Now we have 2D ndarray
    #s += ' final shape:{}'.format(np.shape(a))

    # ??? format string for delimiter
    frm_str = '{:{frm}}'.format(a[0][0], frm=frm)
    frm_str_len = len(frm_str)
    frm_str = '>{}s'.format(frm_str_len)

    s += ' {}'.format(type(a[0][0]))

    rows = 0
    cols = 0
    if isinstance(count, int):               # count=5
        rows = min(count, np.shape(a)[0])
        cols = min(count, np.shape(a)[1])
    if isinstance(count, tuple):             # count=(5,10)
        rows = min(count[0], np.shape(a)[0])
        cols = min(count[1], np.shape(a)[1])

    s += ' print ({},{}) elements'.format(rows, cols)
    # print array info
    print('----\n', s)

    # print array stats
    s =  ' ({:{frm}}/{:{frm}}'.format(np.mean(a), np.mean(np.abs(a)), frm=re.sub('.*\.', '.', _data_format_string(np.mean(np.abs(a)))))
    s += ', {:{frm}}/{:{frm}}'.format(np.std(a), np.std(np.abs(a)),   frm=re.sub('.*\.', '.', _data_format_string(np.std(a))))
    s += ', [{:{frm}}/{:{frm}}]'.format(np.min(a), np.max(a),         frm=re.sub('.*\.', '.', frm))
    s += ' p{}/{}={:{frm}}/{:{frm}})'.format(p1, p99, np.percentile(a, p1), np.percentile(a, p99), frm=re.sub('.*\.', '.', frm))
    print(s, '\n----')

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
                s += '{:{frm}}'.format('...', frm=frm_str)
                #s += ' ... '
            if cols_all:
                s = s[:-frm_str_len]
            s += '\n'

        s += '{:{frm}}\n'.format('...', frm=frm_str)
    if rows_all:
        s = s[:-frm_str_len-1]

    print(s[:-1] + '\n----')

    pass  # print_ndarray()


# test 
if __name__ == "__main__":

    with Profiler('mylib.py testing') as p:

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

    pass  # test section
