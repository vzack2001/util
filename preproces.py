""" create keras preproces model
"""
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

#import warnings
#warnings.filterwarnings('ignore', module='tensorflow')

import functools
import ml_collections
import tf_func # tf helper funcs


def strides_mean_var(x: np.ndarray, w=2, logvar=True, epsilon=1e-9):
    """ x - 2d numpy array shape of (l, m)
        w - moving window width
        return - ndarray shape of (l-w+1, 2)  [i] is (mean, variance) for i-previous w samples
        <<< strides_true_mean_var 100,000 <<<  ET: 9.736 sec.  Mem: rss = 232.38 MB (0.02 MB) / vms = 1021.22 MB (0.00 MB)
    """
    from numpy.lib.stride_tricks import as_strided

    a = np.copy(x, order='C')
    #print(f'strides_mean_var(a[{a.shape}], w={w})', a.strides, x.strides)

    l = np.shape(a)[-0]
    m = np.shape(a)[-1]
    n = a.itemsize

    shape = (l-w+1, m*w,)
    strides = (m*n, n,)    # eq a.strides ?
    b = as_strided(a, shape=shape, strides=strides)
    #print_ndarray(f'b = as_strided(a, shape={shape}, strides={strides})', b, frm='8.3f')

    m = np.mean(b, axis=-1)
    v = np.var(b, axis=-1)

    if logvar:
        v = np.where(v > epsilon, v, epsilon)
        v = np.log(v)

    return np.stack([m, v], axis=-1)

def prepare_data(a: np.ndarray, config: ml_collections.ConfigDict, dtype=np.float32):

    #print_ndarray('a', a, frm='8.3f')
    #  0       1       2      3        4      5      6       7      8      9       10      11      12       13      14      15      16
    #['Open', 'High', 'Low', 'Close', 'AH4', 'AH1', 'AM15', 'AM5', 'AM2', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'LVM1', 'Idx']
    #['Open', 'High', 'Low', 'Close', 'AD1', 'AH4', 'AH1', 'AM15', 'AM5', 'LVW1', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'Idx']

    output_shape = config.output_shape  # (256,17)

    _time = config.p_time  # [1440, 240, 60, 15, 5, 2, 1]                  [7200, 1440, 240, 60, 15, 5, 2]
    _name = config.p_name  # ['D1', 'H4', 'H1', 'M15', 'M5', 'M2', 'M1',]  ['W1', 'D1', 'H4', 'H1', 'M15', 'M5', 'M2',]
    _cols = config.p_cols  # [(None, 9), (4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (None, 15)]

    for name, w, col in zip(_name, _time, _cols):
        mv = strides_mean_var(a, w)

        if w == _time[0]:  # 1440  # init res array
            n = mv.shape[0]
            res = np.zeros((n, output_shape[1]), dtype=dtype)
            res[:,0:4] = a[-n:,0:4]
            res[:,-1] = range(n-1, -1, -1)

        mv = mv[-n:]  # last `n`

        if col[0] is not None:
            res[:,col[0]] = mv[:,0]

        if col[1] is not None:
            res[:,col[1]] = mv[:,1]

    return res


def get_db_bins():
    """ create logvar normalization coeff db
        use in prep()
    """

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

    return db_bins

class Preproces(object):

    @staticmethod
    def build(input_shape, name='preproces', **kwarg):
        """ kwarg = {config: ml_collections.ConfigDict, nbins=15, treshold=0.050, p_range=[5, 95],}:
        """
        #print(f'\n-- Preproces.build(input_shape={input_shape}, name={name}, {kwarg})\n')

        inputs = keras.layers.Input(input_shape, name='input')
        x = inputs

        x = keras.layers.Lambda(prep, trainable=False, arguments=kwarg, name='preproces')(x)

        x = keras.layers.Activation(activation='linear', name='output')(x)

        model = keras.Model(inputs, x, name=name)

        #model.summary()
        #print('model.inputs :', model.inputs)
        #print('model.outputs:', model.outputs)
        #print()

        return model

    pass  # Preproces


def prep(inputs: np.ndarray, config: ml_collections.ConfigDict=None,
        nbins=15,
        p_range=[5,95],
        treshold=0.050,
    ):
    """ kwarg = {config: ml_collections.ConfigDict, nbins=15, treshold=0.050, p_range=[5, 95]}

        prepare data inputs
            from shape (?,256,17)
            to shape (?,16,16,36)

        use tf_func
                _value_range,
                _digitize,
                _histogram2d

    """
    #print(f'\nprep({config.name}):\n', nbins, p_range, treshold)

    #  0       1       2      3        4      5      6       7      8      9       10      11      12      13       14      15      16
    #['Open', 'High', 'Low', 'Close', 'AH4', 'AH1', 'AM15', 'AM5', 'AM2', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'LVM1', 'Idx']
    #['Open', 'High', 'Low', 'Close', 'AD1', 'AH4', 'AH1', 'AM15', 'AM5', 'LVW1', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'Idx']

    _time = config._time              # [256, 64, 16, 16]                       [1024, 256, 64, 16]
    _name = config._name              # ['H4', 'H1', 'M15', 'M5', 'M2', 'D1',]  ['D1', 'H4', 'H1', 'M15', 'M5', 'W1',]

    mean_cols = config.mean_cols      # [4, 5, 6, 7, 8]                         [4, 5, 6, 7, 8]
    logvar_cols = config.logvar_cols  # { 'D1': 9, 'H4': 10, 'H1': 11,  'M15': 12, 'M5': 13, 'M2': 14, 'M1': 15 }
                                      # { 'W1': 9, 'D1': 10, 'H4': 11,  'H1': 12, 'M15': 13, 'M5': 14, 'M2': 15}
    output = []

    db_bins = get_db_bins()

    for i, t in enumerate(_time):
        lv_fast = _name[i]
        lv_slow = _name[i-1]
        name = lv_fast

        with tf.name_scope(name):
            data = inputs[:,-t:,:]
            #print_ndarray(f'{i} {t}: {lv_slow} {lv_fast} / {mean_cols[i]} {mean_cols[i+1]}', data, 0, frm='8.3f')

            # mean section
            ohlc_values = data[...,0:4]         # ['Open', 'High', 'Low', 'Close',]
            price_values = ohlc_values[...,-1]  # 'Close'
            mean_values = tf.reduce_mean(ohlc_values, axis=-1)
            mean_slow = data[...,mean_cols[i]]
            mean_fast = data[...,mean_cols[i+1]]

            value_range = tf_func._value_range(mean_values, percentile=p_range, treshold=treshold)  # (2,?,1)  (min|max,?,1)

            last_value = price_values[:,-1]     # (?,)
            price_range = tf.stack([
                last_value - treshold,
                last_value,
                last_value + treshold,
                ], axis=1)                      # (?,3)

            price_range = tf_func._digitize(price_range, value_range, nbins)       # (?,3)
            price_range = tf.one_hot(price_range, nbins+1, axis=1)                 # (?,16,3)
            price_range = tf.expand_dims(price_range, axis=1)                      # (?,1,16,3)
            price_range = tf.repeat(price_range, nbins+1, axis=1)                  # (?,16,16,3)

            price_values = tf_func._histogram2d(price_values, value_range, nbins)  # (?,16,16,1)
            mean_values = tf_func._histogram2d(mean_values, value_range, nbins)
            mean_slow = tf_func._histogram2d(mean_slow, value_range, nbins)
            mean_fast = tf_func._histogram2d(mean_fast, value_range, nbins)

            # logvar section
            bins = db_bins[lv_fast].to_list()

            logvar_slow = data[..., logvar_cols[lv_slow]]
            logvar_fast = data[..., logvar_cols[lv_fast]]

            logvar_slow = tf_func._histogram2d(logvar_slow, bins, nbins)
            logvar_fast = tf_func._histogram2d(logvar_fast, bins, nbins)

            img = tf.concat([
                price_values,
                mean_values,
                mean_slow,
                mean_fast,
                price_range,
                logvar_slow,
                logvar_fast,
                ], axis=-1)                     # (?,16,16,4+3+2)

            output.append(img)

    output = tf.concat(output, axis=-1)

    return output


def get_prep256_config(data_file='hilo66_logvar_2020.csv', target_file='reward_100_2020.csv'):
    """ Returns `prep` configuration.
        logvar data_file
        ['Date', 'Close1', 'Open', 'High', 'Low', 'Close', 'Spread', 'LVMN1', 'LVW1', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'LVM1', 'AMN1', 'AW1', 'AD1', 'AH4', 'AH1', 'AM15', 'AM5', 'AM2', 'AM1', 'Idx'],
          0       1         2       3       4      5        6         7        8       9       10      11      12       13      14      15      16      17     18     19     20     21      22     23     24     25
    """
    config = ml_collections.ConfigDict()
    config.name = 'prepH4M2_256'

    config.data_file = data_file
    config.target_file = target_file
    #                    0       1        2     3        4      5      6       7      8      9       10      11      12       13      14      15      16
    config.data_list = ['Open', 'High', 'Low', 'Close', 'AH4', 'AH1', 'AM15', 'AM5', 'AM2', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'LVM1', 'Idx']
    config.targets_list = ['noop', 'buy', 'sell', 'done', 'buy_time_100', 'sell_time_100', 'Idx']

    config._time = [256, 64, 16, 16]
    config._name = ['H4', 'H1', 'M15', 'M5', 'M2', 'D1',]
    config.mean_cols = [4, 5, 6, 7, 8]
    config.logvar_cols = { 'D1': 9,  'H4': 10,  'H1': 11,  'M15': 12, 'M5': 13,  'M2': 14,  'M1': 15 }

    config.p_time = [1440, 240, 60, 15, 5, 2, 1]
    config.p_name = ['D1', 'H4', 'H1', 'M15', 'M5', 'M2', 'M1',]
    config.p_cols = [(None, 9), (4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (None, 15)]

    config.output_shape = (256,17)
    config.input_shape = config.output_shape

    print('set config:', config.name)

    return config

def get_prep1024_config(data_file='hilo66_logvar_2020.csv', target_file='reward_100_2020.csv'):
    """ Returns `prep` configuration.
        logvar data_file
        ['Date', 'Close1', 'Open', 'High', 'Low', 'Close', 'Spread', 'LVMN1', 'LVW1', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'LVM1', 'AMN1', 'AW1', 'AD1', 'AH4', 'AH1', 'AM15', 'AM5', 'AM2', 'AM1', 'Idx'],
          0       1         2       3       4      5        6         7        8       9       10      11      12       13      14      15      16      17     18     19     20     21      22     23     24     25
    """
    config = ml_collections.ConfigDict()
    config.name = 'prepD1M5_1024'

    config.data_file = data_file
    config.target_file = target_file
    #                    0       1        2     3        4      5      6       7      8      9       10      11      12       13      14      15      16
    config.data_list = ['Open', 'High', 'Low', 'Close', 'AD1', 'AH4', 'AH1', 'AM15', 'AM5', 'LVW1', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'Idx']
    config.targets_list = ['noop', 'buy', 'sell', 'done', 'buy_time_100', 'sell_time_100', 'class', 'Idx']

    config._time = [1024, 256, 64, 16]
    config._name = ['D1', 'H4', 'H1', 'M15', 'M5', 'W1',]
    config.mean_cols = [4, 5, 6, 7, 8]
    config.logvar_cols = { 'W1': 9, 'D1': 10, 'H4': 11,  'H1': 12,  'M15': 13, 'M5': 14,  'M2': 15}

    config.p_time = [7200, 1440, 240, 60, 15, 5, 2]
    config.p_name = ['W1', 'D1', 'H4', 'H1', 'M15', 'M5', 'M2',]
    config.p_cols = [(None, 9), (4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (None, 15)]

    config.output_shape = (1024,17)
    config.input_shape = config.output_shape

    print('set config:', config.name)

    return config


if __name__ == "__main__":
    from mylib import print_ndarray, data_read_pandas, seq_safe_idx, batch_from_seq

    pd.set_option('display.max_columns', 500)
    #pd.set_option('display.width', 1000)

    def draw_img(image, n_cols=9, n_rows=4):
        import matplotlib.pyplot as plt
        # create a grid of plots
        # https://colorscheme.ru/html-colors.html
        fig, axs = plt.subplots(n_rows,n_cols,figsize=(n_cols,n_rows),facecolor='Gray')

        # plot a sample number into each subplot
        for row in range(n_rows):
            for col in range(n_cols):
                pos = row*n_cols + col
                img = image[:,:,pos]
                #print_ndarray('img[{}]'.format((row,col,pos)), img*255, 16, frm='6.0f')

                # plot image in axes
                axs[row,col].imshow(img, cmap='gray')

                # remove x and y axis
                axs[row,col].axis('off')

        # remove unecessary white space
        plt.tight_layout()

        # display image
        plt.show(block=None)
        pass  # draw_img()

    #tf.compat.v1.disable_eager_execution()
    print('tensorflow version: {0}'.format(tf.__version__))
    print('keras version: {0}'.format(keras.__version__))
    print('tf.executing_eagerly(): {}'.format(tf.executing_eagerly()))

    # define model name and path
    model_name = 'preproces'
    data_path = 'D:/Doc/Pyton/mql/data/'
    model_path = 'D:/Doc/Pyton/util/temp/'

    def get_data(config, data_path='D:/Doc/Pyton/mql/data/'):

        read_csv = functools.partial(
            pd.read_csv,
            parse_dates=['Date'],
            infer_datetime_format=True,
            header=0,
            index_col=0,
            dtype='float32')

        data = data_read_pandas(
            data_file=config.data_file,      # 'hilo66_logvar_2020.csv',
            #target_file='reward_2020.csv',
            target_file=config.target_file,  # 'reward_100_2020.csv',
            data_path=data_path,
            data_list = config.data_list,
            targets_list = config.targets_list,
            read_fn=read_csv,
            )

        return data

    def test_prep(data, config, idx=[28799, 35999], data_seq=0, target_seq=0, target_shift=0):

        data.set_params(data_seq=data_seq, target_seq=target_seq, target_shift=target_shift)
        print(data)

        x, y = data.get_dataset(idx)
        print_ndarray('\nx = data.get_dataset({})'.format(idx), x, 10, frm='8.3f')
        print_ndarray('y = data.get_dataset({})\n'.format(idx), y, 10, frm='8.3f')

        nbins = 15
        p_range = [5,95]
        treshold = 0.050

        if not tf.executing_eagerly():
            data_shape = np.shape(x)[1:]
            model = Preproces.build(data_shape, name=model_name, config=config, nbins=nbins, p_range=p_range, treshold=treshold)
            logdir = 'C:\logs'
            tf.compat.v1.summary.FileWriter(logdir, graph=tf.compat.v1.get_default_graph()).close()
            assert False, 'write graph'

        output = prep(x, config, nbins=nbins, p_range=p_range, treshold=treshold)
        print_ndarray('\noutput\n', output, 0)  # (2,16,16,36)

        return output


    idx = [28799, 35999]
    #data_list = ['Open', 'High', 'Low', 'Close', 'AH4', 'AH1', 'AM15', 'AM5', 'AM2', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'LVM1', 'Idx']
    #targets_list = ['noop', 'buy', 'sell', 'done', 'buy_time_100', 'sell_time_100', 'Idx']  # reward_100_2020.csv
    config256 = get_prep256_config(data_file='hilo66_logvar_2020.csv', target_file='reward_100_2020.csv')
    dr_train = get_data(config256, data_path=data_path)

    output = test_prep(dr_train, config256, idx=idx, data_seq=256, target_seq=0, target_shift=0)
    for image in output:
        draw_img(image, n_cols=9, n_rows=4)

    # test prepare_data()
    x, y = dr_train.get_dataset(idx, data_seq=7200, target_seq=0, target_shift=0)
    print_ndarray('\nx = .get_dataset({})'.format(idx), x, 20, frm='8.3f')

    y = x[0,:,:4]
    a = prepare_data(y, config256)
    print_ndarray(' = .prepare_data({})'.format(np.shape(y)), a, 20, frm='8.3f')


    #data_list = ['Open', 'High', 'Low', 'Close', 'AD1', 'AH4', 'AH1', 'AM15', 'AM5', 'LVW1', 'LVD1', 'LVH4', 'LVH1', 'LVM15', 'LVM5', 'LVM2', 'Idx']
    config1024 = get_prep1024_config(data_file='hilo66_logvar_2020.csv', target_file='reward_100_2020~.csv')
    dr_train = get_data(config1024, data_path=data_path)

    output = test_prep(dr_train, config1024, idx=[28799, 35999], data_seq=1024, target_seq=0, target_shift=0)
    for image in output:
        draw_img(image, n_cols=9, n_rows=4)

    # test prepare_data()
    x, y = dr_train.get_dataset(idx, data_seq=9200, target_seq=0, target_shift=0)
    print_ndarray('\nx = .get_dataset({})'.format(idx), x, 20, frm='8.3f')

    y = x[0,:,:4]
    a = prepare_data(y, config1024)
    print_ndarray(' = .prepare_data({})'.format(np.shape(y)), a, 20, frm='8.3f')

    # test prepare_data()
    idx = dr_train.get_safe_idx(data_seq=0, target_seq=0, target_shift=0)[:2695]
    x, y = dr_train.get_dataset(idx, data_seq=0, target_seq=0, target_shift=0)

    print_ndarray('\nx = dr_train.get_dataset({})'.format((idx[0],idx[-1])), x, 20, frm='7.3f')

    a = x[:2695,:4]  # (1696, 4) (1440+256, 4)

    res = prepare_data(a, config256)
    print_ndarray('res', res, 20, frm='7.3f')

    #seq_safe_idx, batch_from_seq
    idx = seq_safe_idx(res, idx=None, data_seq=256, target_seq=0, target_shift=0)
    print_ndarray(f'idx', idx)
    idx = idx[::-5] #[::-1]
    print_ndarray(f'idx', idx)

    x, y = batch_from_seq(res, res[:,-1:], idx, data_seq=256, target_seq=0, target_shift=0, dtype=np.float32)
    print_ndarray(f'x', x, 20, frm='7.3f')
    print_ndarray('', y)

