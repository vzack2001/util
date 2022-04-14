import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend

from mylib import data_read_pandas


class MeanVarStat(keras.metrics.Metric):

    def __init__(self, name='mean_var_stat', dtype=tf.float64, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.sum2 = self.add_weight(name='sum_squares', initializer='zeros')
        self.min = self.add_weight(name='min', initializer='zeros')
        self.max = self.add_weight(name='max', initializer='zeros')
        self.n = self.add_weight(name='n', initializer='zeros', dtype=tf.int32)
        pass  # __init__()

    def update_state(self, x):
        size = tf.shape(x)[0]
        x = tf.cast(x, dtype=self.dtype)
        self.sum.assign_add(tf.reduce_sum(x))
        self.sum2.assign_add(tf.reduce_sum(tf.square(x)))
        #self.n.assign_add(tf.cast(size, dtype=self.dtype))
        self.n.assign_add(size)
        self.min = min(tf.reduce_min(x), self.min)
        self.max = max(tf.reduce_max(x), self.max)
        pass  # update_state()

    def result(self):
        epsilon = 1e-6
        mean = self.sum / tf.cast(self.n, dtype=self.dtype)
        var = self.sum2 / tf.cast(self.n, dtype=self.dtype) - mean * mean
        return {    'mean'   : mean,
                    'var'    : var,
                    'min'    : self.min,
                    'max'    : self.max,
                    'sum'    : self.sum,
                    'sum2'   : self.sum2,
                    'n'      : self.n,
                }

    def __str__(self):
        r = self.result()
        mean   = r['mean']
        var    = r['var']
        min    = r['min']
        max    = r['max']
        s = f'{mean:.3f} {var:.3f} [{min:.3f} {max:.3f}]'
        return s

    pass  # class MeanVarStat(keras.metrics.Metric)

class ActionRewardStat(keras.metrics.Metric):

    def __init__(self, num_act, name='action_stat', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_act = num_act
        weight_shape = (self.num_act,)
        self.per_action_pos = self.add_weight(name='per_action_pos', shape=weight_shape, initializer='zeros')
        self.per_action_neg = self.add_weight(name='per_action_neg', shape=weight_shape, initializer='zeros')
        self.action_pos = self.add_weight(name='action_pos', shape=weight_shape, initializer='zeros')
        self.action_neg = self.add_weight(name='action_neg', shape=weight_shape, initializer='zeros')
        self.action_sum = self.add_weight(name='action_sum', shape=weight_shape, initializer='zeros')
        pass  # __init__()

    def update_state(self, act_rewards, actions):
        """ act_rewards - numpy.ndarray shape of (?, num_act) - rewards for actions
        """
        policy_actions = tf.squeeze(actions)     # [(?, 1)|(?,)] --> (?,)
        policy_actions_act = tf.one_hot(policy_actions, self.num_act, dtype=tf.float32)          # (?,3)

        action_pos_mask = tf.where(act_rewards > 0, 1.0, 0.0)   # (?,3)
        action_neg_mask = tf.where(act_rewards < 0, 1.0, 0.0)   # (?,3)

        per_action_pos = tf.reduce_sum(action_pos_mask * policy_actions_act, axis=0)  # (3,)
        per_action_neg = tf.reduce_sum(action_neg_mask * policy_actions_act, axis=0)  # (3,)
        action_pos = tf.reduce_sum(action_pos_mask, axis=0)     # (3,)
        action_neg = tf.reduce_sum(action_neg_mask, axis=0)     # (3,)
        action_sum = tf.reduce_sum(policy_actions_act, axis=0)  # (3,)

        self.per_action_pos.assign_add(per_action_pos)
        self.per_action_neg.assign_add(per_action_neg)
        self.action_pos.assign_add(action_pos)
        self.action_neg.assign_add(action_neg)
        self.action_sum.assign_add(action_sum)

        pass  # update_state()

    def result(self):
        epsilon = 1e-6
        total_reward = tf.reduce_sum(self.per_action_pos - self.per_action_neg)
        total_reward = total_reward / tf.reduce_sum(self.per_action_pos + self.per_action_neg + epsilon)
        action_sum = self.action_sum / (tf.reduce_sum(self.action_sum) + epsilon)
        per_action_pos = self.per_action_pos / (self.action_pos + epsilon)
        per_action_neg = self.per_action_neg / (self.action_sum + epsilon)
        return {    'total_reward'   : total_reward,
                    'action_sum'     : action_sum,
                    'per_action_pos' : per_action_pos,
                    'per_action_neg' : per_action_neg,
                }

    def reset_state(self):
        backend.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])
        pass  # reset_state()

    def __str__(self):
        eps = 5e-5

        r = self.result()
        total_reward   = r['total_reward'] * 100.
        action_sum     = r['action_sum'] * 100.
        per_action_pos = r['per_action_pos'] * 100.
        per_action_neg = r['per_action_neg'] * 100.

        #per_action_pos = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_pos[1:]]) + ']'
        #per_action_pos = '[' + ''.join(['{:{frm}}'.format(s, frm=('5.0f' if (100 - np.abs(s)) < eps else '5.1f')) for s in per_action_pos[1:]]) + ']'
        per_action_pos = '[' + ''.join(['{:{frm}}'.format(s, frm=('5.0f' if (100 - np.abs(s)) < eps else '5.1f')) for s in per_action_pos]) + ']'
        per_action_neg = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_neg]) + ']'
        action_sum = '[' + ''.join(['{:5.1f}'.format(s) for s in action_sum]) + ']'
        total_reward = f'{total_reward:6.1f}' if (100 - np.abs(total_reward)) < eps else f'{total_reward:6.2f}'

        return f'{action_sum}{total_reward} {per_action_pos} {per_action_neg}'

    pass  # class ActionRewardStat(keras.metrics.Metric)

class ConditionStat(keras.metrics.Metric):

    def __init__(self, num_act, name='condition_stat', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_act = num_act
        weight_shape = (self.num_act,)
        self.action_sum = self.add_weight(name='action_sum', shape=weight_shape, initializer='zeros')
        self.accepted = self.add_weight(name='accepted', shape=weight_shape, initializer='zeros')
        pass  # __init__()

    def update_state(self, value, condition_str, actions):

        policy_actions = tf.squeeze(actions)     # [(?, 1)|(?,)] --> (?,)
        policy_actions_act = tf.one_hot(policy_actions, self.num_act, dtype=tf.float32)     # (?,3)

        action_sum = tf.reduce_sum(policy_actions_act, axis=0)                # (3,)
        # !!!
        # TODO: remove eval function
        value = eval('value' + condition_str)
        # !!!
        accepted = tf.where(tf.logical_and(policy_actions_act > 0, value), 1.0, 0.0)          # (?,3)
        #accepted = tf.where(value * policy_actions_act > 0, 1.0, 0.0)        # (?,3)
        accepted = tf.reduce_sum(accepted, axis=0)                            # (3,)

        self.action_sum.assign_add(action_sum)
        self.accepted.assign_add(accepted)
        pass  # update_state()

    def result(self):
        epsilon = 1e-6
        action_sum = self.action_sum / tf.reduce_sum(self.action_sum + epsilon)
        accepted = self.accepted / (self.action_sum + epsilon)
        return  {   'accepted'   : accepted,
                    'action_sum' : action_sum,
                }

    def __str__(self):
        r = self.result()
        accepted   = r['accepted'] * 100.
        action_sum = r['action_sum'] * 100.
        action_sum = '[' + ''.join(['{:5.1f}'.format(s) for s in action_sum]) + ']'
        accepted = '[' + ''.join(['{:5.1f}'.format(s) for s in accepted]) + ']'
        return f'{action_sum} {accepted}'

    def reset_state(self):
        backend.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])
        pass  # reset_state()

    pass  # class ConditionStat(keras.metrics.Metric)

class PolicyStat(keras.metrics.Metric):

    def __init__(self, num_act=3, name='policy_stat', **kwargs):

        super().__init__(name=name, **kwargs)  #, trainable=False

        weight_shape = (num_act,)
        self.per_action_pos = self.add_weight(name='per_action_pos', shape=weight_shape, initializer='zeros')
        self.per_action_neg = self.add_weight(name='per_action_neg', shape=weight_shape, initializer='zeros')
        self.action_pos = self.add_weight(name='action_pos', shape=weight_shape, initializer='zeros')
        self.action_neg = self.add_weight(name='action_neg', shape=weight_shape, initializer='zeros')
        self.action_sum = self.add_weight(name='action_sum', shape=weight_shape, initializer='zeros')
        self.overtime = self.add_weight(name='overtime', shape=weight_shape, initializer='zeros')

        pass  # __init__()

    def update_state(self, y_true, actions):
        """
            y_true - numpy.ndarray ['none', 'buy', 'sell', 'done', 'buy_time_100', 'sell_time_100',..., idx]
        """
        act_rewards = y_true[:,:3]               # (?,3) ['none', 'buy', 'sell',]
        done_time = y_true[:,3:6]                # (?,3) ['done', 'buy_time_100', 'sell_time_100',]

        #log_probs = tf.math.log_softmax(tf.ones_like(act_rewards))                   # (?, 3) `random`
        #policy_actions = tf.random.categorical(log_probs, 1, dtype=tf.int32)         # (?, 1)
        #policy_actions = tf.math.argmax(log_probs, axis=1, output_type=tf.int32)     # (?,)

        policy_actions = tf.squeeze(actions)     # [(?, 1)|(?,)] --> (?,)
        policy_actions_act = tf.one_hot(policy_actions, 3, dtype=tf.float32)          # (?,3)

        action_pos_mask = tf.where(act_rewards > 0, 1.0, 0.0)   # (?,3)
        action_neg_mask = tf.where(act_rewards < 0, 1.0, 0.0)   # (?,3)
        overtime = tf.where(done_time * policy_actions_act > 7200, 1.0, 0.0)          # (?,3)

        per_action_pos = tf.reduce_sum(action_pos_mask * policy_actions_act, axis=0)  # (3,)
        per_action_neg = tf.reduce_sum(action_neg_mask * policy_actions_act, axis=0)  # (3,)
        action_pos = tf.reduce_sum(action_pos_mask, axis=0)     # (3,)
        action_neg = tf.reduce_sum(action_neg_mask, axis=0)     # (3,)
        action_sum = tf.reduce_sum(policy_actions_act, axis=0)  # (3,)
        overtime = tf.reduce_sum(overtime, axis=0)              # (3,)

        self.per_action_pos.assign_add(per_action_pos)
        self.per_action_neg.assign_add(per_action_neg)
        self.action_pos.assign_add(action_pos)
        self.action_neg.assign_add(action_neg)
        self.action_sum.assign_add(action_sum)
        self.overtime.assign_add(overtime)

        pass  # update_state()

    def __str__(self):
        eps = 5e-5
        r = self.result()
        total_reward   = r['total_reward'] * 100.
        action_sum     = r['action_sum'] * 100.
        per_action_pos = r['per_action_pos'] * 100.
        per_action_neg = r['per_action_neg'] * 100.
        overtime       = r['overtime'] * 100.

        #per_action_pos = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_pos[1:]]) + ']'
        per_action_pos = '[' + ''.join(['{:{frm}}'.format(s, frm=('5.0f' if (100 - np.abs(s)) < eps else '5.1f')) for s in per_action_pos[1:]]) + ']'
        per_action_neg = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_neg]) + ']'
        action_sum = '[' + ''.join(['{:5.1f}'.format(s) for s in action_sum]) + ']'
        overtime = '[' + ''.join(['{:5.1f}'.format(s) for s in overtime[1:]]) + ']'
        total_reward = f'{total_reward:6.1f}' if (100 - np.abs(total_reward)) < eps else f'{total_reward:6.2f}'
        return f'{action_sum}{total_reward} {per_action_pos} {per_action_neg} {overtime}'

    def result(self):
        epsilon = 1e-6
        total_reward = tf.reduce_sum(self.per_action_pos - self.per_action_neg)
        total_reward = total_reward / tf.reduce_sum(self.per_action_pos + self.per_action_neg + epsilon)
        action_sum = self.action_sum / tf.reduce_sum(self.action_sum + epsilon)
        per_action_pos = self.per_action_pos/(self.action_pos + epsilon)
        per_action_neg = self.per_action_neg/(self.action_sum + epsilon)
        overtime = self.overtime/(self.action_sum + epsilon)
        return {    'total_reward'   : total_reward,
                    'action_sum'     : action_sum,
                    'per_action_pos' : per_action_pos,
                    'per_action_neg' : per_action_neg,
                    'overtime'       : overtime
                }

    def reset_state(self):
        backend.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])
        pass  # reset_state()

    pass  # class PolicyStat(keras.metrics.Metric)

def policy_test(
        test_dataset: data_read_pandas,
        model: keras.Model,
        batch_size=128,
        batch_limit=None,
        prob_size=3,
        verbose=True,
        log_steps=10,
        expected_time=75,
        name=None,
        policy_type='argmax', # 'random', 'ideal', 'categorical'
    ):
    """ Perform a test of the policy.
        Args:
            test_dataset: data_read_pandas test dataset
            model: the actor-critic model
            bath_size: int
        Returns:
            total_reward: obtained score
          0       1      2       3       4               5
        ['noop', 'buy', 'sell', 'done', 'buy_time_100', 'sell_time_100']
    """
    test_batches_it = test_dataset.gen_batch(batch_size=batch_size, shuffle=False, verbose=verbose)

    name =  '' if name is None else name + '_'
    name = name + policy_type
    with Profiler(name, expected_time=expected_time) as p:  # 312/620 T4 # 415/880 K80 # 18 sec. just iterate dataset

        batch_limit = batch_limit or test_dataset.len(batch_size=batch_size) // batch_size
        #print(batch_limit)

        l2 = keras.metrics.Mean()  # log_probs_l2 stat
        kl_stat = keras.metrics.Mean()
        entropy_stat = keras.metrics.Mean()
        policy_stat = PolicyStat(prob_size)

        for i, (x, y) in enumerate(test_batches_it):
            #print_ndarray('x', x, 12, frm='6.0f')  # (?, 256, 17)
            #print_ndarray('y', y, 12, frm='6.0f')  # (?, 7)

            true_labels = y[:,-2]             # (?,)   `class` ['notbuy', 'notsell']
            act_rewards = y[:,:3]             # (?, 3) ['noop', 'buy', 'sell',]
            dones = y[:,3:4]                  # (?, 1)
            #idx = y[:,-1:]                   # (?, 1)

            if policy_type == 'random':
                log_probs = tf.math.log_softmax(tf.ones_like(act_rewards))         # (?, 3) random log_probs
                actions = tf.random.categorical(log_probs, 1, dtype=tf.int32)

            elif policy_type == 'categorical':
                log_probs = tf.math.log_softmax(act_rewards)                       # (?, 3) `true` log_probs
                actions = tf.random.categorical(log_probs, 1, dtype=tf.int32)

            elif policy_type == 'ideal':
                log_probs = tf.math.log_softmax(act_rewards)                       # (?, 3) `true` log_probs
                actions = tf.math.argmax(log_probs, axis=1, output_type=tf.int32)  # (?,)

            else:
                log_probs, values, features = model(x)                             # [(?, 3), (?, 1), ]
                actions = tf.math.argmax(log_probs, axis=1, output_type=tf.int32)  # (?,)

            #print_ndarray('log_probs', log_probs, 0, frm='8.3f')
            #print_ndarray('actions', actions, 0, frm='8.0f')

            # update metrics
            policy_stat.update_state(y, actions)

            log_probs_l2 = tf.reduce_mean(tf.square(log_probs))
            l2.update_state(log_probs_l2)

            #entropy = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.exp(log_probs), log_probs), axis=-1, keepdims=True)
            entropy = -tf.reduce_sum((tf.math.exp(log_probs) * log_probs), axis=-1, keepdims=True)
            entropy = tf.reduce_mean(entropy)
            entropy_stat.update_state(entropy)

            true_log_probs = tf.math.log_softmax(act_rewards)   # (?, 3)
            kl = tf.reduce_sum(tf.math.exp(true_log_probs) * (true_log_probs - log_probs), axis=-1)
            kl = tf.reduce_mean(kl)
            kl_stat.update_state(kl)

            res = f'{policy_stat}  entropy: {entropy_stat.result():5.3f} kl: {kl_stat.result():5.3f} l2:{l2.result():5.2f}'
            if i % log_steps == 0:  #  and i > 0
                if verbose:
                    print(f'{i:4.0f} {res}  {p}')
                pass

            if i >= batch_limit-1:
                break

        if verbose:
            print(f'{i:4.0f} {res}  {p}')

    print(res)

    stat = policy_stat.result() # + {'entropy': entropy_stat.result(), 'kl': kl_stat.result()}

    return stat

if __name__ == "__main__":

    from mylib import Profiler, print_ndarray

    np.random.seed(seed=111)

    def log_softmax(logits: np.ndarray, axis=-1):
        """ tf.math.log_softmax(logits) numpy realization
            logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
        """
        return logits - np.log(np.sum(np.exp(logits), axis=axis, keepdims=True))

    def categorical(probs: np.ndarray, num_samples: int, logits=False, dtype=np.int32):
        """ tf.random.categorical(logits, 1)
        """
        if logits:
            probs = np.exp(log_softmax(probs))

        size = list(probs.shape[:-1]) + [num_samples]
        probs = probs.cumsum(-1).repeat(num_samples, axis=0)
        rand = np.random.uniform(size=size).reshape((-1,1))
        cat = (probs >= rand).argmax(-1)[..., None].reshape(size)
        return np.asarray(cat, dtype=dtype)

    def get_reward_batch(num_act=3, size=12, dtype=np.float32):
        """ get_reward_batch - calculate random reward [0, -1, 1] batch
            with guaranted `0` reward action in each single act.
        """
        reward = [0, -1, 1]
        reward_num = len(reward)
        rewards = np.asarray(reward, dtype=dtype)

        probs = [1/reward_num for _ in range(reward_num)]
        probs = np.ones((size, reward_num), dtype=dtype) * probs
        actions = categorical(probs, num_act, dtype=np.int32)

        probs = [1/num_act for _ in range(num_act)]
        probs = np.ones((size, num_act), dtype=dtype) * probs
        zero_act = categorical(probs, 1, dtype=np.int32)

        indices = np.indices(zero_act.shape)
        actions[indices[0], zero_act] = 0

        return np.asarray(rewards[actions], dtype=dtype)

    def gen_action_reward(probs=[0.6, 0.2, 0.2], batch_size=12, dtype=np.float32):
        num_act = len(probs)
        shape = (batch_size, num_act)

        probs = np.ones(shape, dtype=dtype) * probs

        act_rewards = get_reward_batch(num_act=num_act, size=batch_size, dtype=dtype)

        #actions = categorical(act_rewards, 1, logits=True, dtype=np.int32)
        actions = categorical(probs, 1, dtype=np.int32)

        return actions, act_rewards

    with Profiler('test ActionRewardStat') as p:
        num_act = 4

        stat = ActionRewardStat(num_act)
        print(f'\ntest ActionRewardStat({num_act})')
        print(stat)
        batches = 12
        for i in range(batches):
            actions, act_rewards = gen_action_reward(probs=[0.5, 0.2, 0.2, 0.1], batch_size=8192, dtype=np.float32)
            stat.update_state(act_rewards, actions)
            #print_ndarray('\n1 actions', actions)
            #print_ndarray('\n3 act_rewards = rewards[actions]', act_rewards)
            #print(stat.result())
            print(stat)
        print(stat.result())

        a = []
        stat = MeanVarStat()
        print(f'\ntest MeanVarStat()')
        print(stat)
        for i in range(batches):
            x = np.random.normal(loc=0, scale=1, size=8192)
            a.append(x)
            stat.update_state(x)
            print(stat)
        print(stat.result())
        print(stat)
        print(f'{np.mean(a):.3f}, {np.var(a):.3f}, [{np.min(a):.3f}, {np.max(a):.3f}]')

        stat = ConditionStat(num_act)
        print(f'\ntest ConditionStat({num_act})')
        for i in range(batches):
            actions, act_rewards = gen_action_reward(probs=[0.5, 0.2, 0.2, 0.1], batch_size=8192, dtype=np.float32)
            x = np.random.normal(loc=0, scale=1, size=(8192, num_act))
            stat.update_state(x, '>1', actions)
            print(stat)
        print(stat.result())

    pass
