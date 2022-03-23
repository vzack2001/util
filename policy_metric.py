import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend

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
        r = self.result()
        total_reward   = r['total_reward']
        action_sum     = r['action_sum']
        per_action_pos = r['per_action_pos']
        per_action_neg = r['per_action_neg']
        overtime       = r['overtime']

        # value_if_true if condition else value_if_false
        #per_action_pos = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_pos[1:]]) + ']'
        per_action_pos = '[' + ''.join(["{:{frm}}".format(s, frm=('5.0f' if np.abs(s-100)<0.005 else '5.1f')) for s in per_action_pos[1:]]) + ']'
        per_action_neg = '[' + ''.join(['{:5.1f}'.format(s) for s in per_action_neg]) + ']'
        action_sum = '[' + ''.join(['{:5.1f}'.format(s) for s in action_sum]) + ']'
        overtime = '[' + ''.join(['{:5.1f}'.format(s) for s in overtime[1:]]) + ']'
        return f'{action_sum} {total_reward:5.2f} {per_action_pos} {per_action_neg} {overtime}'

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


