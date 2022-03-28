import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend

from mylib import data_read_pandas, Profiler, print_ndarray

class PolicyStat(keras.metrics.Metric):

    def __init__(self, num_act, name='policy_stat', **kwargs):

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
        total_reward   = r['total_reward']
        action_sum     = r['action_sum']
        per_action_pos = r['per_action_pos']
        per_action_neg = r['per_action_neg']
        overtime       = r['overtime']

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
        total_reward = total_reward / tf.reduce_sum(self.per_action_pos + self.per_action_neg + epsilon) * 100.
        action_sum = self.action_sum / tf.reduce_sum(self.action_sum + epsilon) * 100.
        per_action_pos = self.per_action_pos/(self.action_pos + epsilon) * 100.
        per_action_neg = self.per_action_neg/(self.action_sum + epsilon) * 100.
        overtime = self.overtime/(self.action_sum + epsilon) * 100.
        return {    'total_reward'   : total_reward,
                    'action_sum'     : action_sum,
                    'per_action_pos' : per_action_pos,
                    'per_action_neg' : per_action_neg,
                    'overtime'       : overtime
                }

    def reset_state(self):
        backend.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])
        pass  # reset_state()

    pass  # PolicyStat

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
    pass
